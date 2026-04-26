#!/usr/bin/env python3
"""
Sample random labeling frames from California FRC match videos listed on TBA.

The output is intended for manual labeling, not direct training. Each extracted
image is accompanied by metadata so false positives can be traced back to the
event, match, video, and timestamp.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import yt_dlp


REPO_ROOT = Path(__file__).resolve().parents[1]
TBA_BASE_URL = "https://www.thebluealliance.com/api/v3"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_local_env(path: Path = REPO_ROOT / ".env.local") -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def tba_get(path: str) -> object:
    auth_key = os.environ.get("TBA_AUTH_KEY") or os.environ.get("TBA_API_KEY")
    if not auth_key:
        raise RuntimeError("Set TBA_AUTH_KEY or TBA_API_KEY, or add it to .env.local")

    request = Request(
        f"{TBA_BASE_URL}{path}",
        headers={
            "X-TBA-Auth-Key": auth_key,
            "User-Agent": "ML-Scouter-frame-sampler/1.0",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def safe_slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")[:80] or "unknown"


def match_sort_key(match: dict) -> tuple[int, int, int]:
    comp_order = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}
    return (
        comp_order.get(match.get("comp_level"), 99),
        match.get("set_number") or 0,
        match.get("match_number") or 0,
    )


def match_label(match: dict) -> str:
    comp = str(match.get("comp_level") or "").upper()
    set_number = match.get("set_number") or 1
    match_number = match.get("match_number") or 0
    if comp == "QM":
        return f"qm{match_number}"
    return f"{comp.lower()}{set_number}m{match_number}"


def youtube_url(video_key: str) -> str:
    return f"https://www.youtube.com/watch?v={video_key}"


def select_events(year: int, state_prov: str, max_events: int, rng: random.Random) -> list[dict]:
    events = tba_get(f"/events/{year}/simple")
    ca_events = [
        event
        for event in events
        if str(event.get("country", "")).upper() == "USA"
        and str(event.get("state_prov", "")).upper() == state_prov.upper()
    ]
    rng.shuffle(ca_events)
    return ca_events[:max_events] if max_events else ca_events


def select_video_matches(event_key: str, max_matches: int, rng: random.Random) -> list[dict]:
    matches = tba_get(f"/event/{event_key}/matches")
    video_matches = []
    for match in sorted(matches, key=match_sort_key):
        youtube_videos = [
            video for video in match.get("videos", [])
            if video.get("type") == "youtube" and video.get("key")
        ]
        if youtube_videos:
            match = dict(match)
            match["youtube_key"] = youtube_videos[0]["key"]
            video_matches.append(match)

    rng.shuffle(video_matches)
    return video_matches[:max_matches] if max_matches else video_matches


def video_stream_info(url: str, height_limit: int) -> tuple[str, float | None]:
    format_selector = (
        f"best[ext=mp4][height<={height_limit}]/"
        f"bestvideo[ext=mp4][height<={height_limit}]/"
        f"best[height<={height_limit}]/best"
    )
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "format": format_selector,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    stream_url = info.get("url")
    if not stream_url:
        raise RuntimeError("yt-dlp did not return a playable stream URL")
    return stream_url, info.get("duration")


def read_frame_at(cap: cv2.VideoCapture, second: float) -> object | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ok, frame = cap.read()
    if ok:
        return frame
    return None


def extract_from_stream(
    stream_url: str,
    duration: float | None,
    timestamps: list[float],
) -> list[tuple[float, object]]:
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        return []

    if duration is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if fps and frame_count else None

    frames = []
    for second in timestamps:
        frame = read_frame_at(cap, second)
        if frame is not None:
            frames.append((second, frame))

    cap.release()
    return frames


def download_video(url: str, output_dir: Path) -> Path:
    output_template = str(output_dir / "%(id)s.%(ext)s")
    options = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = Path(ydl.prepare_filename(info))
    if downloaded.suffix.lower() != ".mp4":
        downloaded = downloaded.with_suffix(".mp4")
    if not downloaded.is_file():
        raise FileNotFoundError(f"Downloaded video not found: {downloaded}")
    return downloaded


def sample_timestamps(
    rng: random.Random,
    count: int,
    duration: float | None,
    min_second: float,
    max_second: float | None,
    avoid_last_seconds: float,
) -> list[float]:
    if max_second is None:
        max_second = duration - avoid_last_seconds if duration else 150
    if duration:
        max_second = min(max_second, max(0, duration - avoid_last_seconds))
    min_second = min(min_second, max_second)
    return sorted(rng.uniform(min_second, max_second) for _ in range(count))


def write_frame(path: Path, frame: object, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")


def append_metadata(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "image",
        "event_key",
        "event_name",
        "match_key",
        "match_label",
        "youtube_url",
        "timestamp_seconds",
    ]
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract random frames from California TBA match videos.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--state-prov", default="CA")
    parser.add_argument("--out", default=str(REPO_ROOT / "datasets" / "california_match_frames"))
    parser.add_argument("--max-events", type=int, default=6)
    parser.add_argument("--max-matches-per-event", type=int, default=3)
    parser.add_argument("--frames-per-match", type=int, default=8)
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Stop once the output image folder reaches this many images.",
    )
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--min-second", type=float, default=15.0)
    parser.add_argument("--max-second", type=float, default=None)
    parser.add_argument(
        "--avoid-last-seconds",
        type=float,
        default=15.0,
        help="Do not sample timestamps from this many seconds at the end of each video.",
    )
    parser.add_argument("--height-limit", type=int, default=720)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--download-fallback",
        action="store_true",
        help="Download a temp mp4 if OpenCV cannot read the YouTube stream directly.",
    )
    args = parser.parse_args()

    load_local_env()
    rng = random.Random(args.seed)
    output_root = Path(args.out).expanduser().resolve()
    images_dir = output_root / "images"
    metadata_path = output_root / "metadata.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    existing_count = count_images(images_dir)
    if args.target_count is not None and existing_count >= args.target_count:
        print(f"Output already has {existing_count} images, meeting target {args.target_count}.")
        return

    try:
        events = select_events(args.year, args.state_prov, args.max_events, rng)
    except (HTTPError, URLError, RuntimeError) as error:
        print(f"Could not query TBA events: {error}", file=sys.stderr)
        sys.exit(1)

    if not events:
        print(f"No {args.state_prov} events found for {args.year}.", file=sys.stderr)
        sys.exit(1)

    print(f"Selected {len(events)} {args.state_prov} events for {args.year}.")
    if args.target_count is not None:
        print(f"Existing images: {existing_count}; target: {args.target_count}.")

    rows: list[dict] = []
    extracted = 0
    for event in events:
        if args.target_count is not None and existing_count + extracted >= args.target_count:
            break
        event_key = event["key"]
        event_name = event.get("name") or event_key
        try:
            matches = select_video_matches(event_key, args.max_matches_per_event, rng)
        except (HTTPError, URLError) as error:
            print(f"Skipping {event_key}: could not query matches: {error}")
            continue

        print(f"{event_key}: {len(matches)} sampled matches from {event_name}")
        for match in matches:
            if args.target_count is not None and existing_count + extracted >= args.target_count:
                break
            frame_goal = args.frames_per_match
            if args.target_count is not None:
                frame_goal = min(frame_goal, args.target_count - existing_count - extracted)
            yt_url = youtube_url(match["youtube_key"])
            label = match_label(match)
            try:
                stream_url, duration = video_stream_info(yt_url, args.height_limit)
                timestamps = sample_timestamps(
                    rng,
                    frame_goal,
                    duration,
                    args.min_second,
                    args.max_second,
                    args.avoid_last_seconds,
                )
                frames = extract_from_stream(stream_url, duration, timestamps)
            except Exception as error:
                print(f"  {match['key']}: stream extraction failed: {error}")
                frames = []

            if not frames and args.download_fallback:
                with tempfile.TemporaryDirectory(prefix="ml_scouter_video_") as tmp:
                    try:
                        video_path = download_video(yt_url, Path(tmp))
                        cap = cv2.VideoCapture(str(video_path))
                        fps = cap.get(cv2.CAP_PROP_FPS) or 0
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                        duration = frame_count / fps if fps and frame_count else None
                        timestamps = sample_timestamps(
                            rng,
                            frame_goal,
                            duration,
                            args.min_second,
                            args.max_second,
                            args.avoid_last_seconds,
                        )
                        frames = [
                            (second, frame)
                            for second in timestamps
                            if (frame := read_frame_at(cap, second)) is not None
                        ]
                        cap.release()
                    except Exception as error:
                        print(f"  {match['key']}: download fallback failed: {error}")

            for index, (second, frame) in enumerate(frames, start=1):
                image_name = (
                    f"{event_key}_{label}_{safe_slug(match['key'])}_"
                    f"{int(second * 10):06d}_{index:02d}.jpg"
                )
                image_path = images_dir / image_name
                write_frame(image_path, frame, args.jpeg_quality)
                rows.append(
                    {
                        "image": str(image_path.relative_to(output_root)),
                        "event_key": event_key,
                        "event_name": event_name,
                        "match_key": match["key"],
                        "match_label": label,
                        "youtube_url": yt_url,
                        "timestamp_seconds": f"{second:.2f}",
                    }
                )
                extracted += 1
            print(f"  {match['key']}: wrote {len(frames)} frames")

    append_metadata(metadata_path, rows)
    print()
    print(f"Wrote {extracted} new images to {images_dir}")
    print(f"Total images: {count_images(images_dir)}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
