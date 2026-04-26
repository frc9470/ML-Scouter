"""
A utility for cropping official FIRST Robotics Competition 2026 match videos to 
isolate the horizontal, field-wide perspective.

# Usage
Selecting crop dimensions:
```python crop_top_view.py select \\
  --video /path/to/sample_match.mp4 \\```

The code will automatically store the crop dimensions in `top_view_crop.json`

Reusing crop dimensions across multiple videos:
```python crop_top_view.py crop \\
  --config top_view_crop.json \\
  --glob "/path/to/matches/*.mp4" \\
  --output-dir /path/to/cropped```
"""

import cv2

import argparse
import json
from glob import glob
from pathlib import Path

from utils import center_window, resize_to_fit

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select and apply a reusable crop for the top broadcast view in "
            "FRC match videos."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser(
        "select",
        help="Open a video frame, draw the crop once, and save it as JSON.",
    )
    select_parser.add_argument(
        "--video",
        required=True,
        help="Path to a sample video that uses the target broadcast layout.",
    )
    select_parser.add_argument(
        "--config",
        default="top_view_crop.json",
        help="Where to save the crop configuration JSON.",
    )
    select_parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Zero-based frame index to use for ROI selection.",
    )

    crop_parser = subparsers.add_parser(
        "crop",
        help="Apply a saved crop to one or more videos.",
    )
    crop_parser.add_argument(
        "--config",
        default="top_view_crop.json",
        help="Path to the crop configuration JSON created by the select command.",
    )
    crop_parser.add_argument(
        "--video",
        nargs="+",
        help="One or more explicit video paths to crop.",
    )
    crop_parser.add_argument(
        "--glob",
        dest="glob_pattern",
        help="Glob pattern for batch cropping, for example 'videos/*.mp4'.",
    )
    crop_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for cropped outputs. Defaults to each input video's folder.",
    )
    crop_parser.add_argument(
        "--suffix",
        default="_topview",
        help="Suffix added before the output file extension.",
    )
    crop_parser.add_argument(
        "--codec",
        default="avc1",
        help="FourCC codec for the output video writer.",
    )
    crop_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if they already exist.",
    )

    return parser.parse_args()


def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def read_frame(video_path, frame_index):
    cap = open_video(video_path)
    try:
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(
                f"Could not read frame {frame_index} from video: {video_path}"
            )
        return frame
    finally:
        cap.release()


def save_crop_config(frame, roi, config_path, video_path):
    x, y, w, h = [int(v) for v in roi]
    frame_h, frame_w = frame.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("ROI must have positive width and height.")

    config = {
        "video_path": str(video_path),
        "source_size": {"width": frame_w, "height": frame_h},
        "crop_xywh": {"x": x, "y": y, "w": w, "h": h},
        "crop_normalized": {
            "x": x / frame_w,
            "y": y / frame_h,
            "w": w / frame_w,
            "h": h / frame_h,
        },
    }

    config_path = Path(config_path)
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Saved crop config to {config_path}")
    print(json.dumps(config, indent=2))


def select_crop(video_path, config_path, frame_index):
    frame = read_frame(video_path, frame_index)
    preview, scale = resize_to_fit(frame, return_scale=True)
    instructions = (
        "Draw a rectangle around the top view, then press ENTER or SPACE. "
        "Press C to cancel."
    )
    print(instructions)
    window_name = "Select Top View Crop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, preview.shape[1], preview.shape[0])
    center_window(window_name, preview.shape[1], preview.shape[0])
    roi = cv2.selectROI(
        window_name,
        preview,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyAllWindows()

    if scale <= 0:
        raise ValueError("Scale factor must be positive.")

    scaled_roi = tuple(int(round(value / scale)) for value in roi)
    save_crop_config(frame, scaled_roi, config_path, video_path)


def load_crop_config(config_path):
    config = json.loads(Path(config_path).read_text())
    crop = config["crop_normalized"]
    return crop


def resolve_inputs(video_paths, glob_pattern):
    inputs = []
    if video_paths:
        inputs.extend(video_paths)
    if glob_pattern:
        inputs.extend(sorted(glob(glob_pattern)))

    deduped = []
    seen = set()
    for path in inputs:
        resolved = str(Path(path))
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)

    if not deduped:
        raise ValueError("No input videos provided. Use --video and/or --glob.")
    return deduped


def compute_crop_bounds(norm_crop, frame_w, frame_h):
    x1 = max(0, min(frame_w - 1, round(norm_crop["x"] * frame_w)))
    y1 = max(0, min(frame_h - 1, round(norm_crop["y"] * frame_h)))
    x2 = max(x1 + 1, min(frame_w, round((norm_crop["x"] + norm_crop["w"]) * frame_w)))
    y2 = max(y1 + 1, min(frame_h, round((norm_crop["y"] + norm_crop["h"]) * frame_h)))
    return x1, y1, x2, y2


def build_output_path(input_path, output_dir, suffix):
    input_path = Path(input_path)
    target_dir = Path(output_dir) if output_dir else input_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{input_path.stem}{suffix}{input_path.suffix}"


def crop_video(video_path, output_path, norm_crop, codec, overwrite):
    if output_path.exists() and not overwrite:
        print(f"Skipping existing output: {output_path}")
        return

    cap = open_video(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        x1, y1, x2, y2 = compute_crop_bounds(norm_crop, frame_w, frame_h)
        crop_w = x2 - x1
        crop_h = y2 - y1

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (crop_w, crop_h))
        if not writer.isOpened():
            raise RuntimeError(
                f"Could not open video writer for {output_path}. "
                f"Try a different codec such as 'avc1' or 'mp4v'."
            )

        print(
            f"Cropping {video_path} -> {output_path} "
            f"with box x={x1}, y={y1}, w={crop_w}, h={crop_h}"
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cropped = frame[y1:y2, x1:x2]
            writer.write(cropped)
            frame_idx += 1
            if total_frames and frame_idx % 300 == 0:
                print(f"  processed {frame_idx}/{total_frames} frames")

        writer.release()
        print(f"Finished {output_path}")
    finally:
        cap.release()


def main():
    args = parse_args()

    if args.command == "select":
        select_crop(args.video, args.config, args.frame)
        return

    norm_crop = load_crop_config(args.config)
    inputs = resolve_inputs(args.video, args.glob_pattern)

    for video_path in inputs:
        output_path = build_output_path(video_path, args.output_dir, args.suffix)
        crop_video(video_path, output_path, norm_crop, args.codec, args.overwrite)


if __name__ == "__main__":
    main()
