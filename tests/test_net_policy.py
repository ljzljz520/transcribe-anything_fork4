"""SSRF / network-isolation tests for the daemon (network-media isolation).

All network behavior here is *simulated* (fake resolvers / fake fetchers
/ fake yt-dlp module) — no sockets. Coverage:

* restricted literals and IP obfuscations are denied;
* DNS answers (incl. mixed-record / rebinding names) are all checked;
* the policy spans redirect chains, HLS manifests, key/segment URLs and
  the yt-dlp download subprocess boundary (in-process guarded download);
* restricted destinations are reachable ONLY with admin authorization
  (--allow-host / --allow-cidr, daemon-startup locked);
* multipart uploads keep working and cannot escape the job dir;
* local CLI file input and the CLI subprocess download path keep working
  with no policy attached.
"""

from __future__ import annotations

import email.message
import io
import json
import os
import types
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from transcribe_anything import net_policy
from transcribe_anything.net_policy import (
    NetPolicy,
    NetPolicyError,
    check_url,
    follow_url,
    is_restricted_address,
    parse_ip_literal,
    validate_manifest,
    ytdlp_redirect_guard,
)
from transcribe_anything.server_app import ServerConfig, create_app
from transcribe_anything.ytldp_download import ytdlp_download

DEFAULT_POLICY = NetPolicy()


def _public_resolver(_host: str):
    return ["1.1.1.1"]


@pytest.fixture(autouse=True)
def _simulated_dns(monkeypatch):
    """Hermetic default: every name resolves to a public IP, no real sockets.

    Individual tests override this inside their own monkeypatch when they
    need an internal answer.
    """
    monkeypatch.setattr(net_policy, "default_resolver", _public_resolver)


# --------------------------------------------------------------------------
# restricted address classification and obfuscations
# --------------------------------------------------------------------------

DEFAULT_BLOCKED = {
    "127.0.0.1",
    "127.1",
    "127.0.0.2",
    "10.0.0.5",
    "10.255.255.255",
    "172.16.4.4",
    "172.31.0.1",
    "192.168.1.1",
    "169.254.169.254",  # cloud metadata
    "169.254.1.1",
    "100.64.0.1",  # CGNAT
    "0.0.0.0",
    "255.255.255.255",
    "224.0.0.1",
    "[::1]",
    "[fe80::1]",
    "[fc00::7]",
    "[ff02::1]",
    "[::ffff:127.0.0.1]",
    "[::ffff:a9fe:a9fe]",
    "2130706433",  # decimal 127.0.0.1
    "0x7f000001",  # hex
    "017700000001",  # octal
    "0177.0.0.1",  # dotted octal
    "0x7f.0x0.0x0.0x1",
    "127.0.1",  # inet_aton 3-part
    "192.168.044.01",
}


@pytest.mark.parametrize("host", sorted(DEFAULT_BLOCKED))
def test_restricted_literals_denied(host: str) -> None:
    addr = parse_ip_literal(host)
    assert addr is not None, f"{host!r} should parse as an IP literal"
    assert is_restricted_address(addr), f"{host!r} ({addr}) must be classified restricted"
    with pytest.raises(NetPolicyError):
        check_url(f"http://{host}/secret", DEFAULT_POLICY, resolver=_public_resolver)


PUBLIC_LITERALS = {"1.1.1.1", "8.8.8.8", "93.184.216.34", "172.32.0.1", "172.15.0.1", "100.63.0.1", "100.128.0.1", "[2606:4700::1111]"}


@pytest.mark.parametrize("host", sorted(PUBLIC_LITERALS))
def test_public_literals_allowed(host: str) -> None:
    check_url(f"https://{host}/a.mp3", DEFAULT_POLICY, resolver=_public_resolver)


def test_non_literals_are_not_parsed_as_ip() -> None:
    assert parse_ip_literal("example.com") is None
    assert parse_ip_literal("1.1.1.1.in-addr.arpa") is None


# --------------------------------------------------------------------------
# scheme / shape rejection — local paths and non-http schemes
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "/etc/passwd",
        "../../etc/passwd",
        "C:\\Users\\admin\\secret.txt",
        "file:///etc/passwd",
        "file://127.0.0.1/etc/passwd",
        "ftp://example.com/x.mp3",
        "gopher://127.0.0.1/",
        "dict://localhost/",
        "http://",
        "http://127.0.0.1\r\nX-Evil: 1/x",
        "http://127.0.0.1\\@example.com/x",
        "",
    ],
)
def test_non_http_and_local_path_urls_rejected(url: str) -> None:
    with pytest.raises(NetPolicyError):
        check_url(url, DEFAULT_POLICY, resolver=_public_resolver)


# --------------------------------------------------------------------------
# simulated DNS: every answer must be acceptable
# --------------------------------------------------------------------------


def test_name_resolving_to_internal_rejected() -> None:
    with pytest.raises(NetPolicyError):
        check_url("http://rebind.evil.test/a.mp3", DEFAULT_POLICY, resolver=lambda _h: ["169.254.169.254"])


def test_mixed_records_one_internal_rejected() -> None:
    # One public A record cannot smuggle one internal record past the check.
    with pytest.raises(NetPolicyError):
        check_url("https://cdn.evil.test/a.mp3", DEFAULT_POLICY, resolver=lambda _h: ["1.1.1.1", "10.0.0.9"])


def test_localhost_alias_blocked_even_without_dns() -> None:
    called = []
    with pytest.raises(NetPolicyError):
        check_url("http://localhost:8080/admin", DEFAULT_POLICY, resolver=lambda h: called.append(h) or ["1.1.1.1"])
    assert called == []  # short-circuited before DNS


def test_public_name_allowed() -> None:
    check_url("https://media.example.com/audio.m4a", DEFAULT_POLICY, resolver=_public_resolver)


def test_unresolvable_name_rejected(monkeypatch) -> None:
    import socket

    def _fail(_host):
        raise socket.gaierror("no such host")

    with pytest.raises(NetPolicyError):
        check_url("https://nope.example.com/x", DEFAULT_POLICY, resolver=_fail)


# --------------------------------------------------------------------------
# admin explicit authorization
# --------------------------------------------------------------------------


def test_cidr_authorization_opens_internal_range_only() -> None:
    policy = NetPolicy(allow_cidrs=["127.0.0.0/8", "169.254.0.0/16"])
    check_url("http://127.0.0.1:9999/x", policy, resolver=_public_resolver)
    check_url("http://169.254.169.254/latest/meta-data/", policy, resolver=_public_resolver)
    with pytest.raises(NetPolicyError):
        check_url("http://10.0.0.1/x", policy, resolver=_public_resolver)


def test_host_authorization_bypasses_resolution_for_named_destination() -> None:
    policy = NetPolicy(allow_hosts=["media.internal.example"])
    # Even though the name resolves to an internal IP, the admin explicitly
    # authorized this host.
    check_url("http://media.internal.example/a.m3u8", policy, resolver=lambda _h: ["10.1.2.3"])
    with pytest.raises(NetPolicyError):
        check_url("http://other.internal.example/a.m3u8", policy, resolver=lambda _h: ["10.1.2.3"])


def test_host_suffix_authorization() -> None:
    policy = NetPolicy(allow_hosts=[".corp.example"])
    check_url("https://edge.corp.example/x", policy, resolver=lambda _h: ["10.2.3.4"])
    check_url("https://corp.example/x", policy, resolver=lambda _h: ["10.2.3.4"])
    with pytest.raises(NetPolicyError):
        check_url("https://evilcorp.example/x", policy, resolver=lambda _h: ["10.2.3.4"])


def test_invalid_admin_authorization_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        NetPolicy(allow_cidrs=["not-a-cidr"])
    with pytest.raises(ValueError):
        NetPolicy(allow_hosts=["https://with-scheme.example"])


# --------------------------------------------------------------------------
# redirect chains — policy must apply at every hop
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int = 200, location: str | None = None):
        self.status_code = status_code
        self.headers = {"location": location} if location else {}


def test_redirect_to_internal_is_blocked_before_connect() -> None:
    hits: list[str] = []

    def fetch_once(url: str):
        hits.append(url)
        if len(hits) == 1:
            return _FakeResponse(302, "http://127.0.0.1:9/admin")
        return _FakeResponse(200)

    with pytest.raises(NetPolicyError):
        follow_url("https://public.example.com/redirect", DEFAULT_POLICY, fetch_once, resolver=_public_resolver)
    assert hits == ["https://public.example.com/redirect"]  # never connected to 127.0.0.1


def test_redirect_relative_location_resolved_and_checked() -> None:
    def fetch_once(url: str):
        return _FakeResponse(302, "http://169.254.169.254/meta-data/") if url.endswith("/redirect") else _FakeResponse(200)

    with pytest.raises(NetPolicyError):
        follow_url("https://public.example.com/a/redirect", DEFAULT_POLICY, fetch_once, resolver=_public_resolver)


def test_redirect_chain_to_public_succeeds_and_authorized_internal_hop_allowed() -> None:
    sequence = iter([_FakeResponse(302, "https://cdn.example.net/b"), _FakeResponse(307, "http://127.0.0.1/c"), _FakeResponse(200)])

    def fetch_once(_url: str):
        return next(sequence)

    policy = NetPolicy(allow_cidrs=["127.0.0.0/8"])
    response = follow_url("https://a.example.com/start", policy, fetch_once, resolver=_public_resolver)
    assert response.status_code == 200


def test_redirect_loop_rejected() -> None:
    def fetch_once(url: str):
        target = "https://b.example.com/y" if url.endswith("/x") else "https://b.example.com/x"
        return _FakeResponse(302, target)

    with pytest.raises(NetPolicyError):
        follow_url("https://b.example.com/x", DEFAULT_POLICY, fetch_once, resolver=_public_resolver)


# --------------------------------------------------------------------------
# HLS manifests / segments / keys
# --------------------------------------------------------------------------

MASTER_PLAYLIST = """
#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1280000
https://media.example.com/low/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2560000
http://169.254.169.254/high/index.m3u8
"""

MEDIA_PLAYLIST = """
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-KEY:METHOD=AES-128,URI="http://127.0.0.1/keys/enc.key"
seg1.ts
https://media.example.com/seg2.ts
"""

CLEAN_PLAYLIST = """
#EXTM3U
#EXT-X-KEY:METHOD=AES-128,URI="https://media.example.com/key.bin"
seg1.ts
https://media.example.com/seg2.ts
"""


def test_manifest_variant_to_metadata_ip_rejected() -> None:
    with pytest.raises(NetPolicyError):
        validate_manifest(MASTER_PLAYLIST, "https://media.example.com/master.m3u8", DEFAULT_POLICY, resolver=_public_resolver)


def test_manifest_segment_and_key_urls_checked() -> None:
    # 1) relative segment resolves against the public base -> allowed
    validate_manifest(CLEAN_PLAYLIST, "https://media.example.com/idx.m3u8", DEFAULT_POLICY, resolver=_public_resolver)
    # 2) embedded AES key pointing at loopback -> rejected
    with pytest.raises(NetPolicyError):
        validate_manifest(MEDIA_PLAYLIST, "https://media.example.com/idx.m3u8", DEFAULT_POLICY, resolver=_public_resolver)


def test_manifest_internal_segments_allowed_only_with_authorization() -> None:
    policy = NetPolicy(allow_cidrs=["127.0.0.0/8", "169.254.0.0/16"])
    validate_manifest(MEDIA_PLAYLIST, "http://169.254.169.254/idx.m3u8", policy, resolver=_public_resolver)


# --------------------------------------------------------------------------
# yt-dlp guarded download (fake module simulating entry/manifest/segments)
# --------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, url: str):
        self.url = url


class _FakeYoutubeDL:
    """Simulates yt_dlp.YoutubeDL surface the guard relies on."""

    manifest_urls: list[str] = []

    def __init__(self, params):
        self.params = params
        self.requested: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def urlopen(self, req):
        self.requested.append(req.url)
        return None

    def download(self, urls):
        for url in urls:
            self.urlopen(_FakeRequest(url))
        for url in self.manifest_urls:
            self.urlopen(_FakeRequest(url))
        out_name = self.params["outtmpl"] % {"ext": "mp4"}
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        with open(out_name, "wb") as handle:
            handle.write(b"fake media")


def _fake_ytdlp_module():
    module = types.ModuleType("yt_dlp")
    module.YoutubeDL = _FakeYoutubeDL
    networking = types.ModuleType("yt_dlp.networking")
    urllib_mod = types.ModuleType("yt_dlp.networking._urllib")

    class FakeRedirectHandler(urllib.request.HTTPRedirectHandler):
        pass

    urllib_mod.RedirectHandler = FakeRedirectHandler
    networking._urllib = urllib_mod
    module.networking = networking
    return module


def test_guarded_download_all_public(tmp_path) -> None:
    fake = _fake_ytdlp_module()
    _FakeYoutubeDL.manifest_urls = [
        "https://media.example.com/a.m3u8",
        "https://media.example.com/seg1.ts",
        "https://media.example.com/seg2.ts",
    ]
    out = ytdlp_download("https://media.example.com/watch?v=1", str(tmp_path), DEFAULT_POLICY, ytdlp=fake, resolver=_public_resolver)
    assert os.path.isfile(out)
    assert out.startswith(str(tmp_path))


def test_guarded_download_blocks_internal_segment(tmp_path) -> None:
    fake = _fake_ytdlp_module()
    _FakeYoutubeDL.manifest_urls = [
        "https://media.example.com/a.m3u8",
        "http://169.254.169.254/seg1.ts",
    ]
    with pytest.raises(NetPolicyError):
        ytdlp_download("https://media.example.com/watch?v=2", str(tmp_path), DEFAULT_POLICY, ytdlp=fake, resolver=_public_resolver)
    # nothing downloaded survives the blocked job
    assert not list(tmp_path.iterdir())


def test_guarded_download_blocks_internal_entry_url(tmp_path) -> None:
    fake = _fake_ytdlp_module()
    with pytest.raises(NetPolicyError):
        ytdlp_download("http://127.0.0.1:5000/video", str(tmp_path), DEFAULT_POLICY, ytdlp=fake)


def test_guarded_download_respects_authorized_cidr(tmp_path) -> None:
    fake = _fake_ytdlp_module()
    policy = NetPolicy(allow_cidrs=["169.254.0.0/16"])
    _FakeYoutubeDL.manifest_urls = ["http://169.254.169.254/seg1.ts"]
    out = ytdlp_download("http://169.254.169.254/v", str(tmp_path), policy, ytdlp=fake, resolver=_public_resolver)
    assert os.path.isfile(out)


def test_guarded_download_fails_closed_without_redirect_hook(tmp_path) -> None:
    fake = _fake_ytdlp_module()
    del fake.networking._urllib.RedirectHandler
    with pytest.raises(NetPolicyError):
        ytdlp_download("https://media.example.com/x", str(tmp_path), DEFAULT_POLICY, ytdlp=fake, resolver=_public_resolver)


def test_redirect_hook_blocks_internal_hop() -> None:
    fake = _fake_ytdlp_module()
    handler_cls = fake.networking._urllib.RedirectHandler
    req = urllib.request.Request("https://media.example.com/a.m3u8")
    headers = email.message_from_string("Location: http://127.0.0.1/secret\n")
    with ytdlp_redirect_guard(fake, DEFAULT_POLICY, resolver=_public_resolver):
        handler = handler_cls()
        with pytest.raises(NetPolicyError):
            handler.http_error_302(req, io.BytesIO(b""), 302, "Found", headers)
    # after the context exits, the original hook is restored (no global leak)
    assert "check" not in handler_cls.redirect_request.__name__


def test_redirect_hook_allows_public_hop_and_repairs_afterward() -> None:
    fake = _fake_ytdlp_module()
    handler_cls = fake.networking._urllib.RedirectHandler
    req = urllib.request.Request("https://media.example.com/a.m3u8")
    headers = email.message_from_string("Location: https://cdn.example.net/b\n")
    with ytdlp_redirect_guard(fake, DEFAULT_POLICY, resolver=_public_resolver):
        handler = handler_cls()
        # Drive just redirect_request (http_error_302 on newer Pythons
        # also touches Request.timeout, which is irrelevant to the guard).
        new_req = handler.redirect_request(req, io.BytesIO(b""), 302, "Found", headers, "https://cdn.example.net/b")
        assert new_req is not None
        assert new_req.full_url == "https://cdn.example.net/b"
    # after the context exits, the original hook is restored (no global leak)
    assert "check" not in handler_cls.redirect_request.__name__


def test_cli_subprocess_download_uses_argv_list_without_shell(tmp_path, monkeypatch) -> None:
    """Local CLI path preserved: subprocess still runs, but no shell interpolation."""
    from transcribe_anything import ytldp_download as ytdlp_mod

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["shell"] = kwargs.get("shell")
        captured["cwd"] = kwargs.get("cwd")
        # emulate yt-dlp output
        with open(os.path.join(kwargs["cwd"], "out.mp3"), "wb") as handle:
            handle.write(b"x")

    monkeypatch.setattr(ytdlp_mod.subprocess, "run", fake_run)
    malicious_url = "http://example.com/x; touch /tmp/pwned_should_not_exist"
    out = ytdlp_download(malicious_url, str(tmp_path), None)
    assert os.path.isfile(out)
    assert captured["shell"] is False
    assert isinstance(captured["cmd"], list)
    # the whole URL is one argv element — shell metacharacters inert
    assert captured["cmd"][-3] == malicious_url
    assert "/tmp/pwned_should_not_exist" not in captured["cmd"]


# --------------------------------------------------------------------------
# daemon HTTP integration
# --------------------------------------------------------------------------


def _make_client(tmp_path, **config_kwargs):
    captured = {}

    def fake_transcribe(*, url_or_file, output_dir, **kwargs):
        captured["url_or_file"] = url_or_file
        captured["output_dir"] = output_dir
        captured.update(kwargs)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "out.txt").write_text("ok", encoding="utf-8")
        return str(out)

    cfg = ServerConfig(host="127.0.0.1", model="tiny", job_root=str(tmp_path / "jobs"), **config_kwargs)
    app = create_app(cfg, transcribe_fn=fake_transcribe)
    return TestClient(app), captured


def _wait(client, job_id):
    import time

    for _ in range(100):
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError("job did not terminate")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8765/v1/jobs",
        "http://localhost:80/admin",
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://[::1]/admin",
        "http://10.0.0.5/internal",
        "http://2130706433/secret",  # decimal-obfuscated loopback
        "file:///etc/passwd",
        "/etc/passwd",
        "ftp://example.com/x.mp3",
    ],
)
def test_daemon_rejects_restricted_and_local_urls(tmp_path, url) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        resp = client.post("/v1/transcribe", json={"url": url})
    assert resp.status_code == 400, resp.text
    assert "policy" in resp.json()["detail"].lower()


def test_daemon_rejects_dns_rebinding_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(net_policy, "default_resolver", lambda _h: ["127.0.0.1"])
    client, _ = _make_client(tmp_path)
    with client:
        resp = client.post("/v1/transcribe", json={"url": "https://rebind.evil.test/x"})
    assert resp.status_code == 400


def test_daemon_rejects_client_internal_path_overrides(tmp_path) -> None:
    client, _ = _make_client(tmp_path)
    with client:
        for key in ("output_dir", "out_dir", "job_root", "artifact_dir", "input", "path", "output_path"):
            resp = client.post("/v1/transcribe", json={"url": "https://media.example.com/x", key: "/etc"})
            assert resp.status_code == 400, (key, resp.text)
            assert "server-internal" in resp.json()["detail"]


def test_daemon_accepts_public_url_and_passes_policy_to_worker(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(net_policy, "default_resolver", _public_resolver)
    client, captured = _make_client(tmp_path)
    with client:
        resp = client.post("/v1/transcribe", json={"url": "https://media.example.com/x.mp3", "language": "en"})
        assert resp.status_code == 202, resp.text
        final = _wait(client, resp.json()["job_id"])
        assert final["status"] == "completed"
    assert captured["url_or_file"] == "https://media.example.com/x.mp3"
    assert isinstance(captured["net_policy"], NetPolicy)


def test_daemon_authorized_cidr_allows_internal_url(tmp_path) -> None:
    client, captured = _make_client(tmp_path, network_allow_cidrs=["127.0.0.0/8"])
    with client:
        blocked = client.post("/v1/transcribe", json={"url": "http://10.1.1.1/x"})
        allowed = client.post("/v1/transcribe", json={"url": "http://127.0.0.1:9999/x"})
    assert blocked.status_code == 400
    assert allowed.status_code == 202, allowed.text
    assert captured["url_or_file"] == "http://127.0.0.1:9999/x"


def test_daemon_authorized_host_allows_internal_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(net_policy, "default_resolver", lambda _h: ["10.5.5.5"])
    client, _ = _make_client(tmp_path, network_allow_hosts=["approved.media.corp"])
    with client:
        denied = client.post("/v1/transcribe", json={"url": "http://other.media.corp/x"})
        accepted = client.post("/v1/transcribe", json={"url": "http://approved.media.corp/x"})
    assert denied.status_code == 400
    assert accepted.status_code == 202, accepted.text


def test_normal_multipart_upload_still_works(tmp_path) -> None:
    client, captured = _make_client(tmp_path)
    with client:
        resp = client.post(
            "/v1/transcribe",
            files={"file": ("audio.wav", io.BytesIO(b"RIFFxxxxWAVE"), "audio/wav")},
            data={"options": json.dumps({"language": "en"})},
        )
        assert resp.status_code == 202, resp.text
        final = _wait(client, resp.json()["job_id"])
        assert final["status"] == "completed"
    stored = Path(captured["url_or_file"])
    assert stored.is_file()
    assert stored.name == "audio.wav"
    # quarantine: upload input lives under the per-job artifact directory
    assert stored.is_relative_to(Path(captured["output_dir"]))


def test_upload_filename_traversal_is_neutralized(tmp_path) -> None:
    client, captured = _make_client(tmp_path)
    evil_names = ["../../../../tmp/pwned.bin", "..\\..\\windows\\system32\\pwned.bin", "\x00../evil.bin", "./.hidden.wav"]
    with client:
        for index, name in enumerate(evil_names):
            resp = client.post(
                "/v1/transcribe",
                files={"file": (name, io.BytesIO(b"abc"), "application/octet-stream")},
            )
            assert resp.status_code == 202, (name, resp.text)
            _wait(client, resp.json()["job_id"])
            stored = Path(captured["url_or_file"])
            job_root = tmp_path / "jobs"
            assert stored.is_relative_to(job_root.resolve()), (name, stored)
            assert ".." not in stored.parts
            assert not stored.name.startswith(".")
            assert not Path("/tmp/pwned.bin").exists()
            assert stored.read_bytes() == b"abc"


def test_webhook_internal_literal_rejected_at_submit(tmp_path) -> None:
    client, _ = _make_client(tmp_path, allow_webhooks=True)
    with client:
        resp = client.post(
            "/v1/transcribe",
            json={"url": "https://media.example.com/x", "webhook_url": "http://127.0.0.1:9/hook"},
        )
    assert resp.status_code == 400
    assert "webhook" in resp.json()["detail"]


def test_capabilities_reports_admin_authorizations(tmp_path) -> None:
    client, _ = _make_client(tmp_path, network_allow_hosts=["h.example"], network_allow_cidrs=["10.0.0.0/8"])
    with client:
        body = client.get("/v1/capabilities").json()
    assert body["network_allow_hosts"] == ["h.example"]
    assert body["network_allow_cidrs"] == ["10.0.0.0/8"]


# --------------------------------------------------------------------------
# config + CLI plumbing
# --------------------------------------------------------------------------


def test_config_rejects_bad_authorization_at_validate() -> None:
    cfg = ServerConfig(network_allow_cidrs=["nonsense"])
    with pytest.raises(ValueError):
        cfg.validate()
    cfg = ServerConfig(network_allow_hosts=["no spaces allowed"])
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_env_round_trip_preserves_authorizations() -> None:
    from transcribe_anything.server_config import config_from_env, config_to_env

    cfg = ServerConfig(network_allow_hosts=["a.example", ".b.example"], network_allow_cidrs=["127.0.0.0/8"])
    rebuilt = config_from_env(config_to_env(cfg))
    assert rebuilt.network_allow_hosts == ["a.example", ".b.example"]
    assert rebuilt.network_allow_cidrs == ["127.0.0.0/8"]


def test_cli_flags_flow_into_config() -> None:
    from transcribe_anything.cli_serve import _config_from_args, _parse_args

    args = _parse_args(["--allow-host", "media.corp", "--allow-host", ".cdn.corp", "--allow-cidr", "10.0.0.0/8"])
    cfg = _config_from_args(args)
    assert cfg.network_allow_hosts == ["media.corp", ".cdn.corp"]
    assert cfg.network_allow_cidrs == ["10.0.0.0/8"]
    cfg.validate()


# --------------------------------------------------------------------------
# local CLI file input is unaffected by the daemon policy
# --------------------------------------------------------------------------


def test_local_cli_input_never_enters_policy_guarded_download(tmp_path, monkeypatch) -> None:
    """fetch_audio(local path) without a policy skips yt-dlp entirely."""
    from transcribe_anything import audio

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake mp4")
    out_wav = tmp_path / "out.wav"

    def _ytdlp_should_not_run(*_a, **_kw):
        raise AssertionError("local CLI input must not invoke the network downloader")

    monkeypatch.setattr(audio, "ytdlp_download", _ytdlp_should_not_run)
    monkeypatch.setattr(audio.shutil, "which", lambda _name: "/usr/bin/static_ffmpeg")

    def fake_run(cmd, **kwargs):
        cwd = kwargs.get("cwd")
        if cwd:
            with open(os.path.join(cwd, "out.wav"), "wb") as handle:
                handle.write(b"RIFF")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    audio.fetch_audio(str(src), str(out_wav))
    assert out_wav.read_bytes() == b"RIFF"


def test_api_guard_allows_only_quarantined_local_input(tmp_path, monkeypatch) -> None:
    """With a daemon policy, local input must resolve inside the artifact dir."""
    from transcribe_anything import api

    # Avoid the static_ffmpeg bootstrap (irrelevant to the containment check).
    monkeypatch.setattr(api, "_add_static_ffmpeg_paths", lambda: None)

    artifact_dir = tmp_path / "jobdir"
    artifact_dir.mkdir()
    uploaded = artifact_dir / "_input" / "audio.wav"
    uploaded.parent.mkdir()
    uploaded.write_bytes(b"x")
    outsider = tmp_path / "secret.wav"
    outsider.write_bytes(b"x")

    seen = {}

    def fake_fetch(url_or_file, out_wav, net_policy=None):
        seen["url_or_file"] = url_or_file
        seen["net_policy"] = net_policy
        raise RuntimeError("stop-after-fetch")

    monkeypatch.setattr(api, "fetch_audio", fake_fetch)
    policy = NetPolicy()

    # outsider is refused before ffmpeg/the backend touches it
    with pytest.raises(ValueError, match="job directory"):
        api.transcribe(str(outsider), output_dir=str(artifact_dir), model="tiny", device="cpu", net_policy=policy)
    assert "url_or_file" not in seen

    # quarantined upload passes the guard (then stops at the patched fetch)
    with pytest.raises(RuntimeError, match="stop-after-fetch"):
        api.transcribe(str(uploaded), output_dir=str(artifact_dir), model="tiny", device="cpu", net_policy=policy)
    assert seen["url_or_file"] == str(uploaded)
    assert seen["net_policy"] is policy


def test_api_without_policy_accepts_any_local_file_like_cli(tmp_path, monkeypatch) -> None:
    """Local CLI / library call: no policy, no containment restriction."""
    from transcribe_anything import api

    monkeypatch.setattr(api, "_add_static_ffmpeg_paths", lambda: None)
    local = tmp_path / "video.mp4"
    local.write_bytes(b"x")
    seen = {}

    def fake_fetch(url_or_file, out_wav, net_policy=None):
        seen["called"] = True
        seen["net_policy"] = net_policy
        raise RuntimeError("stop-after-fetch")

    monkeypatch.setattr(api, "fetch_audio", fake_fetch)
    with pytest.raises(RuntimeError, match="stop-after-fetch"):
        api.transcribe(str(local), output_dir=str(tmp_path / "text"), model="tiny", device="cpu")
    assert seen["called"] is True
    assert seen["net_policy"] is None
