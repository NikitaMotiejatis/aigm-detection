# /// script
# requires-python = ">=3.10"
# dependencies = ["yt-dlp", "imageio-ffmpeg"]
# ///
"""Download the real MusicCaps clips (the human-made class for FakeMusicCaps).

MusicCaps is published only as a list of YouTube IDs with start/end times, so the
10-second clips have to be cut from YouTube. Clips whose videos were removed or
are region-blocked are skipped and listed in <out>/failed.csv.

Usage:
    uv run scripts/fetch_musiccaps.py --out data/raw/musiccaps
    uv run scripts/fetch_musiccaps.py --out data/raw/musiccaps --limit 5   # smoke test
    uv run scripts/fetch_musiccaps.py --cookies cookies.txt --workers 2 --sleep 2   # if YouTube asks "confirm you're not a bot"

Files are named <ytid>.wav, the same naming FakeMusicCaps uses for generated clips.
ffmpeg is taken from the imageio-ffmpeg package, no system install is needed.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import yt_dlp
from yt_dlp.utils import download_range_func

CSV_URL = "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv"


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(CSV_URL) as resp:
            csv_path.write_bytes(resp.read())
    with io.open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ffmpeg_dir() -> str:
    """yt-dlp only accepts a binary called 'ffmpeg', so link the bundled one under that name."""
    exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    link_dir = Path.home() / ".cache" / "aigm-detection" / "ffmpeg"
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / "ffmpeg"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(exe)
    # yt-dlp checks PATH (not ffmpeg_location) before allowing partial downloads
    os.environ["PATH"] = str(link_dir) + os.pathsep + os.environ.get("PATH", "")
    return str(link_dir)


def fetch(row: dict, out: Path, sample_rate: int, ffmpeg: str, auth: dict) -> tuple[str, str]:
    ytid = row["ytid"]
    target = out / f"{ytid}.wav"
    if target.exists() and target.stat().st_size > 0:
        return ytid, "exists"
    start, end = float(row["start_s"]), float(row["end_s"])
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "format": "bestaudio/best",
        "outtmpl": str(out / f"{ytid}.%(ext)s"),
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "ffmpeg_location": ffmpeg,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": {"extractaudio": ["-ar", str(sample_rate)]},
        "retries": 3,
        **auth,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={ytid}"])
    except Exception as exc:  # removed, private or blocked video
        return ytid, f"failed: {str(exc).splitlines()[0][:200]}"
    return ytid, "ok" if target.exists() else "failed: no output file"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/raw/musiccaps"))
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--cookies", type=Path, default=None,
                    help="Netscape-format cookies.txt exported from a logged-in YouTube session")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="read cookies straight from a browser, e.g. firefox or chrome")
    ap.add_argument("--sleep", type=float, default=0,
                    help="seconds to wait between requests, which lowers the chance of a bot check")
    args = ap.parse_args()

    auth: dict = {}
    if args.cookies:
        auth["cookiefile"] = str(args.cookies)
    if args.cookies_from_browser:
        auth["cookiesfrombrowser"] = (args.cookies_from_browser,)
    if args.sleep:
        auth["sleep_interval_requests"] = args.sleep
        auth["sleep_interval"] = args.sleep

    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.out / "musiccaps-public.csv")
    if args.limit:
        rows = rows[: args.limit]

    ffmpeg = ffmpeg_dir()
    counts = {"ok": 0, "exists": 0, "failed": 0}
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = pool.map(lambda r: fetch(r, args.out, args.sample_rate, ffmpeg, auth), rows)
        for i, (ytid, status) in enumerate(jobs, 1):
            key = status.split(":")[0]
            counts[key] += 1
            if key == "failed":
                failed.append((ytid, status))
            if i % 50 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] {counts}", flush=True)

    with open(args.out / "failed.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("ytid", "reason"), *failed])
    print(f"done: {counts}; failures listed in {args.out / 'failed.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
