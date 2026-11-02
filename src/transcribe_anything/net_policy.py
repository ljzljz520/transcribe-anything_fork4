"""
SSRF / egress policy for daemon-originated network media fetches.

Remote clients of the transcribe-anything daemon can ask it to fetch a
network URL. That makes the daemon an SSRF target: a malicious media URL
(or a redirect / HLS manifest / segment it points at) could otherwise
make the daemon talk to loopback services, link-local cloud metadata
endpoints (``169.254.169.254``), or RFC1918 internals.

This module is deliberately FastAPI-free and stdlib-only so both the
host-side launcher and the daemon iso-env can import it. The pieces:

* :class:`NetPolicy` — admin-locked egress configuration (default deny
  for restricted address ranges; explicit host/CIDR authorizations).
* :func:`check_url` — validates a single URL: scheme allowlist, host
  allowlist, IP-literal parsing (incl. decimal/hex/octal/``inet_aton``
  obfuscations), and DNS resolution with *every* answer checked.
* :func:`follow_url` — redirect-aware fetch loop: the policy is applied
  to the entry URL and to every ``Location`` hop (relative hops are
  joined the same way HTTP clients do).
* :func:`validate_manifest` — scans an HLS media playlist / master
  manifest (and ``URI="..."`` attributes such as ``EXT-X-KEY``) so
  segment and key URLs are checked before anything requests them.
* :func:`make_guarded_youtube_dl` / :func:`ytdlp_redirect_guard` —
  install the same policy on yt-dlp's in-process downloader, which is
  what performs the entry fetch, manifest fetch, AES-key fetch and
  every fragment/segment GET.

Every network-touching seam accepts an injectable ``resolver`` so the
policy can be exercised against a *simulated* network in unit tests.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional
from urllib.parse import urljoin, urlsplit

# A resolver maps a hostname to the IP literal strings it resolves to.
Resolver = Callable[[str], Iterable[str]]

# HTTP statuses that carry a Location-based redirect.
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

# Hostnames libc resolves to loopback without (or in addition to) DNS.
_LOCAL_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "ip6-localnet",
        "broadcasthost",
    }
)


class NetPolicyError(ValueError):
    """Raised when a URL/address violates the daemon's egress policy."""


# ---------------------------------------------------------------------------
# Restricted-address classification
# ---------------------------------------------------------------------------

_BLOCKED_V4_NETWORKS = tuple(
    ipaddress.ip_network(net)
    for net in (
        "0.0.0.0/8",  # "this" network
        "10.0.0.0/8",  # RFC1918
        "100.64.0.0/10",  # CGNAT
        "127.0.0.0/8",  # loopback
        "169.254.0.0/16",  # link-local + cloud metadata
        "172.16.0.0/12",  # RFC1918
        "192.0.0.0/24",  # IETF protocol assignments
        "192.0.2.0/24",  # TEST-NET-1
        "192.88.99.0/24",  # 6to4 relay anycast (deprecated)
        "192.168.0.0/16",  # RFC1918
        "198.18.0.0/15",  # benchmarking
        "198.51.100.0/24",  # TEST-NET-2
        "203.0.113.0/24",  # TEST-NET-3
        "224.0.0.0/4",  # multicast
        "240.0.0.0/4",  # reserved + limited broadcast
    )
)

_BLOCKED_V6_NETWORKS = tuple(
    ipaddress.ip_network(net)
    for net in (
        "::/128",  # unspecified
        "::1/128",  # loopback
        "::/96",  # deprecated IPv4-compatible IPv6
        "::ffff:0:0/96",  # IPv4-mapped (unwrapped below; blocked raw)
        "64:ff9b:1::/48",  # RFC8215 local-use NAT64
        "fe80::/10",  # link-local
        "fc00::/7",  # unique local
        "ff00::/8",  # multicast
        "2001::/23",  # IETF protocol assignments (incl. Teredo)
        "2001:10::/28",  # ORCHID
        "2001:db8::/32",  # documentation
    )
)


def _unwrap_embedded_v4(addr: ipaddress._BaseAddress) -> ipaddress._BaseAddress:
    """Reduce IPv6 forms that tunnel an IPv4 address to that IPv4 address."""
    if isinstance(addr, ipaddress.IPv6Address):
        mapped = addr.ipv4_mapped
        if mapped is not None:
            return mapped
        packed = addr.packed
        # ::a.b.c.d (IPv4-compatible, deprecated but still accepted by
        # parsers) and the well-known NAT64 prefix 64:ff9b::/96 both embed
        # a routable IPv4 address in the low 32 bits.
        if packed[:12] == b"\x00" * 12 and addr not in (ipaddress.IPv6Address("::"), ipaddress.IPv6Address("::1")):
            return ipaddress.IPv4Address(packed[12:])
        if packed[:8] == b"\x00\x64\xff\x9b" + b"\x00" * 4:
            # 64:ff9b:0000:0000:0000:0000:<v4>
            return ipaddress.IPv4Address(packed[12:])
    return addr


def is_restricted_address(addr: ipaddress._BaseAddress) -> bool:
    """True if ``addr`` is loopback/private/link-local/reserved (denied)."""
    addr = _unwrap_embedded_v4(addr)
    if isinstance(addr, ipaddress.IPv4Address):
        return any(addr in net for net in _BLOCKED_V4_NETWORKS)
    return any(addr in net for net in _BLOCKED_V6_NETWORKS)


def _parse_scalar(part: str) -> Optional[int]:
    """Parse one dotted component using C ``inet_aton`` base rules."""
    if not part:
        return None
    lowered = part.lower()
    try:
        if lowered.startswith("0x"):
            digits = lowered[2:]
            if not digits or any(c not in "0123456789abcdef" for c in digits):
                return None
            return int(digits, 16)
        if len(lowered) > 1 and lowered.startswith("0") and all(c in "01234567" for c in lowered):
            return int(lowered, 8)
        if lowered.isdigit():
            return int(lowered, 10)
    except ValueError:
        return None
    return None


def parse_ip_literal(host: str) -> Optional[ipaddress._BaseAddress]:
    """Parse an IP literal, accepting common SSRF obfuscations.

    Handles bracketed IPv6, scoped IPv6 (``%iface``), IPv4-mapped IPv6,
    and ``inet_aton`` forms: single 32-bit integers in decimal/hex/octal
    (``2130706433``, ``0x7f000001``, ``017700000001``) and 2-4 part
    addresses with hex/octal parts (``0177.0.0.1``, ``127.1``).

    Returns ``None`` when ``host`` is not an IP literal (i.e. a DNS name).
    """
    if not host:
        return None
    text = host.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    text = text.split("%", 1)[0]
    try:
        return _unwrap_embedded_v4(ipaddress.ip_address(text))
    except ValueError:
        pass
    # 0 dots: single 32-bit scalar (2130706433 / 0x7f000001 / 017700000001).
    # Ordinary DNS names fall through to _parse_scalar, which rejects letters.
    if ":" in text or not 0 <= text.count(".") <= 3:
        return None
    parts = text.split(".")
    numbers: list[int] = []
    for part in parts:
        value = _parse_scalar(part)
        if value is None or value < 0:
            return None
        numbers.append(value)
    total = 0
    for index, value in enumerate(numbers):
        is_last = index == len(numbers) - 1
        width_bytes = (4 - len(numbers) + 1) if is_last else 1
        if value >= 1 << (8 * width_bytes):
            return None
        # inet_aton packs every non-last part as one byte at a fixed
        # top-of-word slot (part 0 -> 24, 1 -> 16, 2 -> 8); the last
        # part fills whatever width remains (32/24/16/8 bits).
        shift = 0 if is_last else 8 * (3 - index)
        total |= value << shift
    if not 0 <= total <= 0xFFFFFFFF:
        return None
    return ipaddress.IPv4Address(total)


def default_resolver(host: str) -> list[str]:
    """Default DNS resolver: every A/AAAA answer getaddrinfo returns."""
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise NetPolicyError(f"host {host!r} could not be resolved") from exc
    except OSError as exc:  # pragma: no cover - resolver failure parity
        raise NetPolicyError(f"host {host!r} could not be resolved") from exc
    return list({info[4][0] for info in infos})


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class NetPolicy:
    """Admin-locked egress policy.

    ``allow_host`` / ``allow_cidrs`` are the *only* way restricted
    destinations become reachable; they are daemon-startup settings
    (``--allow-host`` / ``--allow-cidr``), never client-controllable.
    """

    allow_hosts: Iterable[str] = field(default_factory=tuple)
    allow_cidrs: Iterable[str] = field(default_factory=tuple)
    allowed_schemes: tuple[str, ...] = ("http", "https")
    max_redirects: int = 10

    def __post_init__(self) -> None:
        hosts = tuple(str(h).lower().rstrip(".") for h in self.allow_hosts)
        for host in hosts:
            if not host or any(c in host for c in "/?#@: \t"):
                raise ValueError(f"invalid --allow-host entry: {host!r}")
        self.allow_hosts = hosts  # type: ignore[assignment]
        self.allow_cidrs = tuple(self.allow_cidrs)  # type: ignore[assignment]
        self._allow_networks = [ipaddress.ip_network(cidr, strict=False) for cidr in self.allow_cidrs]
        self.allowed_schemes = tuple(s.lower() for s in self.allowed_schemes)

    def host_authorized(self, hostname: str) -> bool:
        host = hostname.lower().rstrip(".")
        if host in self.allow_hosts:
            return True
        # An entry like ".internal.example.com" authorizes the apex and
        # every sub-domain (suffix match on a label boundary).
        return any(host == entry[1:] or host.endswith(entry) for entry in self.allow_hosts if entry.startswith("."))

    def ip_authorized(self, addr: ipaddress._BaseAddress) -> bool:
        addr = _unwrap_embedded_v4(addr)
        if not is_restricted_address(addr):
            return True
        return any(addr in network for network in self._allow_networks)


def check_url(
    url: str,
    policy: NetPolicy,
    *,
    resolver: Optional[Resolver] = None,
    resolve_names: bool = True,
) -> list[ipaddress._BaseAddress]:
    """Validate one URL against ``policy``; return the resolved addresses.

    Raises :class:`NetPolicyError` on any violation. ALL DNS answers are
    checked (a name with one internal answer is rejected, which also
    catches DNS-rebinding style multi-record payloads).

    ``resolve_names=False`` performs the static subset only (schemes,
    literals, hostname allowlist) for early request validation before
    the daemon spends a DNS lookup; DNS names are re-checked fully at
    fetch time.
    """
    if not isinstance(url, str) or not url:
        raise NetPolicyError("url must be a non-empty string")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url) or "\\" in url:
        raise NetPolicyError("url contains illegal characters")
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise NetPolicyError("malformed url") from exc
    scheme = parts.scheme.lower()
    if scheme not in policy.allowed_schemes:
        allowed = ", ".join(policy.allowed_schemes)
        raise NetPolicyError(f"url scheme {scheme!r} is not permitted (allowed: {allowed})")
    try:
        hostname = parts.hostname
    except ValueError as exc:
        raise NetPolicyError("malformed url (bad host)") from exc
    if not hostname:
        raise NetPolicyError("url must include a host")
    hostname = hostname.lower().rstrip(".")
    if hostname in _LOCAL_HOSTNAMES:
        if not policy.host_authorized(hostname):
            raise NetPolicyError(f"host {hostname!r} is a restricted (loopback) address")
        return []
    if policy.host_authorized(hostname):
        return []
    literal = parse_ip_literal(hostname)
    if literal is not None:
        if not policy.ip_authorized(literal):
            raise NetPolicyError("url resolves to a restricted network address")
        return [literal]
    if not resolve_names:
        return []
    resolve = resolver if resolver is not None else default_resolver
    try:
        answers = list(resolve(hostname))
    except NetPolicyError:
        raise
    except (socket.gaierror, OSError) as exc:
        raise NetPolicyError(f"host {hostname!r} could not be resolved") from exc
    if not answers:
        raise NetPolicyError(f"host {hostname!r} did not resolve to any address")
    addresses: list[ipaddress._BaseAddress] = []
    for answer in answers:
        try:
            addr = _unwrap_embedded_v4(ipaddress.ip_address(str(answer)))
        except ValueError as exc:
            raise NetPolicyError(f"host {hostname!r} returned an unparseable address") from exc
        if not policy.ip_authorized(addr):
            raise NetPolicyError(f"host {hostname!r} resolves to a restricted network address")
        addresses.append(addr)
    return addresses


# ---------------------------------------------------------------------------
# Redirect-aware fetching (policy on every hop)
# ---------------------------------------------------------------------------


def _response_status(response: Any) -> Optional[int]:
    return getattr(response, "status_code", None) or getattr(response, "status", None)


def _response_location(response: Any) -> Optional[str]:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    if hasattr(headers, "get"):
        return headers.get("location") or headers.get("Location")
    return None


def follow_url(
    url: str,
    policy: NetPolicy,
    fetch_once: Callable[[str], Any],
    *,
    resolver: Optional[Resolver] = None,
) -> Any:
    """Drive a request through redirects, validating *every* hop.

    ``fetch_once(current_url)`` performs a single, non-redirecting
    request (redirect following must be disabled on the underlying
    client) and returns an object exposing ``status_code``/``status``
    and ``headers['Location']`` for redirect responses. The policy is
    enforced before each request, so an open-redirect to an internal
    address is blocked before the daemon connects.
    """
    current = url
    seen: set[str] = set()
    hops = 0
    while True:
        check_url(current, policy, resolver=resolver)
        if current in seen:
            raise NetPolicyError("redirect loop detected")
        seen.add(current)
        response = fetch_once(current)
        status = _response_status(response)
        location = _response_location(response)
        if status in REDIRECT_STATUSES and location:
            hops += 1
            if hops > policy.max_redirects:
                raise NetPolicyError(f"more than {policy.max_redirects} redirects")
            current = urljoin(current, location)
            continue
        return response


# ---------------------------------------------------------------------------
# HLS / DASH-adjacent manifest scanning
# ---------------------------------------------------------------------------

_M3U8_URI_ATTR = re.compile(r'URI="([^"]+)"', re.IGNORECASE)


def iter_manifest_urls(manifest: str | bytes, base_url: str) -> Iterator[str]:
    """Yield every URL referenced by an HLS manifest, resolved to absolute.

    Covers plain segment / variant-playlist lines and ``URI="..."``
    attributes on tags such as ``#EXT-X-KEY`` (AES keys), ``#EXT-X-MAP``
    (init segments) and ``#EXT-X-MEDIA``.
    """
    if isinstance(manifest, bytes):
        manifest = manifest.decode("utf-8", "replace")
    for raw_line in manifest.splitlines():
        line = raw_line.strip().lstrip("﻿")
        if not line:
            continue
        if line.startswith("#"):
            for referenced in _M3U8_URI_ATTR.findall(line):
                yield urljoin(base_url, referenced.strip())
        else:
            yield urljoin(base_url, line)


def validate_manifest(
    manifest: str | bytes,
    base_url: str,
    policy: NetPolicy,
    *,
    resolver: Optional[Resolver] = None,
) -> None:
    """Reject an HLS manifest that references any restricted URL."""
    for referenced in iter_manifest_urls(manifest, base_url):
        check_url(referenced, policy, resolver=resolver)


# ---------------------------------------------------------------------------
# yt-dlp in-process guard
# ---------------------------------------------------------------------------


def make_guarded_youtube_dl(ytdlp: Any, policy: NetPolicy, *, resolver: Optional[Resolver] = None):
    """Return a ``YoutubeDL`` subclass that policy-checks every request.

    yt-dlp fetches the entry page, every HLS/DASH manifest, every
    AES key and every fragment/segment through ``YoutubeDL.urlopen``,
    so a single override puts the policy in front of all of them.
    Redirect hops are covered separately by
    :func:`ytdlp_redirect_guard`, which patches the RH-level redirect
    handler urllib uses between ``urlopen`` calls.
    """

    class GuardedYoutubeDL(ytdlp.YoutubeDL):  # type: ignore[misc]
        def urlopen(self, req: Any):  # type: ignore[override]
            if isinstance(req, str):
                target = req
            else:
                target = getattr(req, "url", None) or getattr(req, "full_url", None)
            check_url(target, policy, resolver=resolver)
            return super().urlopen(req)

    return GuardedYoutubeDL


@contextmanager
def ytdlp_redirect_guard(ytdlp: Any, policy: NetPolicy, *, resolver: Optional[Resolver] = None) -> Iterator[None]:
    """Install redirect-hop policy checks on yt-dlp's urllib RH.

    Fail closed: if the expected internal hook is missing (unsupported
    yt-dlp version) callers must refuse the fetch rather than silently
    running without redirect protection.
    """
    networking = getattr(ytdlp, "networking", None)
    urllib_mod = getattr(networking, "_urllib", None)
    handler_cls = getattr(urllib_mod, "RedirectHandler", None)
    if handler_cls is None or not hasattr(handler_cls, "redirect_request"):
        raise NetPolicyError(
            "cannot install the yt-dlp redirect policy hook (unsupported yt-dlp internals); "
            "refusing to fetch the untrusted URL"
        )
    original = handler_cls.redirect_request

    def checked_redirect(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        check_url(urljoin(req.get_full_url(), newurl), policy, resolver=resolver)
        return original(self, req, fp, code, msg, headers, newurl)

    handler_cls.redirect_request = checked_redirect
    try:
        yield
    finally:
        handler_cls.redirect_request = original
