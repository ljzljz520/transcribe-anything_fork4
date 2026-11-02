"""
Download utility for yt-dlp.

Two distinct trust domains share this helper:

* **Daemon network fetch** (``policy is not None``): the URL came from a
  remote client. The download runs **in-process** through a guarded
  :class:`yt_dlp.YoutubeDL` whose entry/manifest/key/segment requests and
  redirect hops all pass the daemon's :class:`NetPolicy`. If yt-dlp
  cannot be imported the fetch fails closed instead of falling back to
  an unguarded subprocess.
* **Local CLI fetch** (``policy is None``): the URL was supplied by the
  user running the CLI on their own machine — single-user trust. The
  historical subprocess behavior is preserved, but the URL is now a
  separate argv element (``shell=False``) instead of being interpolated
  into a shell string, so shell metacharacters in a URL can no longer
  execute commands.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from transcribe_anything.net_policy import (
    NetPolicy,
    check_url,
    make_guarded_youtube_dl,
    ytdlp_redirect_guard,
)
from transcribe_anything.util import PROCESS_TIMEOUT

# Suffixes yt-dlp leaves behind while/after a failed or partial download.
_INCOMPLETE_SUFFIXES = (".part", ".ytdl", ".tmp", ".frag")
_OUTPUT_TEMPLATE = "out.%(ext)s"


def _empty_dir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for entry in os.listdir(outdir):
        os.remove(os.path.join(outdir, entry))


def _find_downloaded_file(outdir: str) -> str:
    candidates = sorted(
        str(path)
        for path in Path(outdir).iterdir()
        if path.is_file() and path.name.startswith("out.") and not path.name.endswith(_INCOMPLETE_SUFFIXES)
    )
    assert len(candidates) == 1, f"Expected 1 downloaded file, got {candidates}"
    assert os.path.exists(candidates[0]), f"The expected file {candidates[0]} doesn't exist"
    return candidates[0]


def _cli_subprocess_download(url: str, outdir: str) -> str:
    """Local-CLI path: argv list (no shell), historical flags preserved."""
    cmd_list = [
        "yt-dlp",
        "--no-check-certificate",
        "-x",
        url,
        "-o",
        _OUTPUT_TEMPLATE,
    ]
    print(f"Running:\n  {subprocess.list2cmdline(cmd_list)}")
    subprocess.run(
        cmd_list,
        shell=False,
        cwd=outdir,
        check=True,
        timeout=PROCESS_TIMEOUT,
        universal_newlines=True,
    )
    return _find_downloaded_file(outdir)


def _guarded_download(
    url: str,
    outdir: str,
    policy: NetPolicy,
    *,
    ytdlp: Any = None,
    resolver=None,
) -> str:
    """Daemon path: in-process yt-dlp with the egress policy enforced."""
    if ytdlp is None:
        try:
            import yt_dlp as ytdlp  # type: ignore
        except ImportError as exc:
            # Fail closed: an unguarded subprocess could not enforce the
            # policy on redirects, manifests or segments.
            raise RuntimeError(
                "yt_dlp is required to fetch client-supplied URLs inside the daemon"
            ) from exc

    # Entry URL (covers the case where extractor does no network at all).
    check_url(url, policy, resolver=resolver)

    params = {
        "outtmpl": os.path.join(outdir, _OUTPUT_TEMPLATE),
        "format": "bestaudio/best",
        "noplaylist": True,
        # TLS verification stays ON for untrusted URLs (the CLI subprocess
        # used --no-check-certificates; a MITM must not be able to steer the
        # daemon around the egress policy).
        "nocheckcertificate": False,
        "noprogress": True,
        "quiet": False,
        "overwrites": True,
        "retries": 3,
    }
    guarded_cls = make_guarded_youtube_dl(ytdlp, policy, resolver=resolver)
    # Every urlopen() (entry page, playlists/manifests, AES keys and each
    # fragment/segment) is checked by the subclass; every redirect hop is
    # checked by the RH-level guard installed below.
    with ytdlp_redirect_guard(ytdlp, policy, resolver=resolver):
        with guarded_cls(params) as downloader:
            downloader.download([url])
    return _find_downloaded_file(outdir)


def ytdlp_download(
    url: str,
    outdir: str,
    policy: Optional[NetPolicy] = None,
    *,
    ytdlp: Any = None,
    resolver=None,
) -> str:
    """Downloads ``url`` into ``outdir`` and returns the resulting file path.

    When ``policy`` is provided (daemon mode) all network access is
    constrained to it; when omitted (local CLI) the historical
    subprocess downloader is used, without a shell.
    """
    _empty_dir(outdir)
    if policy is not None:
        return _guarded_download(url, outdir, policy, ytdlp=ytdlp, resolver=resolver)
    return _cli_subprocess_download(url, outdir)
