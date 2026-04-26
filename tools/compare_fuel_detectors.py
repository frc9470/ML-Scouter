#!/usr/bin/env python3
"""
Generate quick visual comparisons for candidate FUEL detectors.

This is intentionally a review tool, not an evaluator: the sampled frames are
unlabeled, so the output is a set of annotated contact sheets and detection
counts for human inspection.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Detection:
    xyxy: tuple[int, int, int, int]
    score: float
    label: str


@dataclass
class CandidateCrop:
    image_path: Path
    image: np.ndarray
    detection: Detection


def list_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    )


def orange_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Wide enough to include washed-out yellow/orange FUEL, but not red bumpers.
    lower = np.array([10, 55, 80], dtype=np.uint8)
    upper = np.array([42, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def play_area_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    keep = np.zeros_like(mask)
    y1 = int(h * 0.10)
    y2 = int(h * 0.88)
    x1 = int(w * 0.02)
    x2 = int(w * 0.98)
    keep[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return keep


def box_in_play_area(box: tuple[int, int, int, int], width: int, height: int) -> bool:
    x1, y1, x2, y2 = clamp_box(box, width, height)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return width * 0.02 <= cx <= width * 0.98 and height * 0.10 <= cy <= height * 0.88


def box_color_fraction(image: np.ndarray, box: tuple[int, int, int, int]) -> float:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    mask = orange_mask(image[y1:y2, x1:x2])
    return float(cv2.countNonZero(mask)) / float(mask.size)


def yolo_detections(model: YOLO, image: np.ndarray, conf: float) -> list[Detection]:
    result = model.predict(image, verbose=False, imgsz=960, conf=conf)[0]
    detections: list[Detection] = []
    if result.boxes is None:
        return detections
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    names = model.names or {}
    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        detections.append(
            Detection(
                xyxy=(x1, y1, x2, y2),
                score=float(score),
                label=str(names.get(int(cls_id), cls_id)),
            )
        )
    return detections


def yolo_shape_filter(image: np.ndarray, detections: list[Detection]) -> list[Detection]:
    h, w = image.shape[:2]
    image_area = h * w
    kept: list[Detection] = []
    for det in detections:
        x1, y1, x2, y2 = clamp_box(det.xyxy, w, h)
        if not box_in_play_area((x1, y1, x2, y2), w, h):
            continue
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 3 or bh <= 3:
            continue
        area = bw * bh
        area_fraction = area / image_area
        aspect = bw / bh
        color_fraction = box_color_fraction(image, (x1, y1, x2, y2))

        # FUEL should be compact. These bounds intentionally reject large
        # orange/yellow field structures and wide robot panels.
        if not (0.000015 <= area_fraction <= 0.018):
            continue
        if not (0.45 <= aspect <= 2.2):
            continue
        if color_fraction < 0.035:
            continue
        kept.append(Detection((x1, y1, x2, y2), det.score, f"{det.label} filtered"))
    return kept


def hsv_contour_detections(image: np.ndarray) -> list[Detection]:
    h, w = image.shape[:2]
    image_area = h * w
    mask = play_area_mask(orange_mask(image))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 8:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        box_area = bw * bh
        if box_area <= 0:
            continue
        area_fraction = box_area / image_area
        aspect = bw / bh
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0
        fill = area / box_area
        if not (0.00001 <= area_fraction <= 0.012):
            continue
        if not (0.45 <= aspect <= 2.1):
            continue
        if circularity < 0.22 and fill < 0.32:
            continue
        score = float(min(1.0, 0.45 * circularity + 0.55 * fill))
        detections.append(Detection((x, y, x + bw, y + bh), score, "hsv"))
    detections.sort(key=lambda det: det.score, reverse=True)
    return detections[:30]


def hough_circle_detections(image: np.ndarray) -> list[Detection]:
    h, w = image.shape[:2]
    mask = play_area_mask(orange_mask(image))
    blurred = cv2.GaussianBlur(mask, (9, 9), 1.8)
    min_radius = max(3, int(min(w, h) * 0.004))
    max_radius = max(8, int(min(w, h) * 0.055))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.4,
        minDist=max(8, min_radius * 2),
        param1=80,
        param2=10,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    detections: list[Detection] = []
    for cx, cy, radius in np.round(circles[0]).astype(int):
        x1, y1, x2, y2 = cx - radius, cy - radius, cx + radius, cy + radius
        x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), w, h)
        if not box_in_play_area((x1, y1, x2, y2), w, h):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        color_fraction = box_color_fraction(image, (x1, y1, x2, y2))
        if color_fraction < 0.08:
            continue
        detections.append(Detection((x1, y1, x2, y2), color_fraction, "circle"))
    detections.sort(key=lambda det: det.score, reverse=True)
    return detections[:30]


def sam3_status() -> tuple[bool, str]:
    if importlib.util.find_spec("sam3") is None:
        return False, "SAM 3 package is not installed."
    if not torch.cuda.is_available():
        return False, "SAM 3 package may be installed, but no CUDA GPU is available."
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        return False, "SAM 3 checkpoints require authenticated Hugging Face access."
    return True, "SAM 3 appears runnable; this script does not auto-run the 848M checkpoint yet."


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    title: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{det.score:.2f}"
        cv2.putText(canvas, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def make_tile(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    tile = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    tile[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return tile


def make_text_tile(lines: list[str], size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    tile = np.full((target_h, target_w, 3), 238, dtype=np.uint8)
    y = 28
    for line in lines:
        for wrapped in textwrap.wrap(line, width=33) or [""]:
            cv2.putText(tile, wrapped, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1, cv2.LINE_AA)
            y += 20
            if y > target_h - 12:
                return tile
    return tile


def contact_sheet(rows: list[list[np.ndarray]], out_path: Path, tile_size: tuple[int, int]) -> None:
    rendered_rows = []
    for row in rows:
        rendered_rows.append(np.hstack([make_tile(tile, tile_size) for tile in row]))
    sheet = np.vstack(rendered_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"Could not write {out_path}")


def crop_with_context(image: np.ndarray, box: tuple[int, int, int, int], pad: float = 1.2) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(bw, bh) * (1 + pad)
    nx1 = int(cx - side / 2)
    ny1 = int(cy - side / 2)
    nx2 = int(cx + side / 2)
    ny2 = int(cy + side / 2)
    nx1, ny1, nx2, ny2 = clamp_box((nx1, ny1, nx2, ny2), w, h)
    crop = image[ny1:ny2, nx1:nx2].copy()
    if crop.size == 0:
        return image.copy()
    cv2.rectangle(crop, (x1 - nx1, y1 - ny1), (x2 - nx1, y2 - ny1), (0, 255, 255), 2)
    return crop


def candidate_sheet(candidates: list[CandidateCrop], out_path: Path, limit: int = 80) -> None:
    candidates = sorted(candidates, key=lambda item: item.detection.score, reverse=True)[:limit]
    if not candidates:
        sheet = make_text_tile(["No candidates"], (240, 160))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), sheet)
        return

    cols = 8
    tile_size = (180, 150)
    tiles = []
    for item in candidates:
        tile = make_tile(crop_with_context(item.image, item.detection.xyxy), tile_size)
        label = f"{item.detection.score:.2f} {item.image_path.stem[:18]}"
        cv2.putText(tile, label, (6, tile_size[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)

    rows = []
    for index in range(0, len(tiles), cols):
        row = tiles[index:index + cols]
        while len(row) < cols:
            row.append(np.full((tile_size[1], tile_size[0], 3), 245, dtype=np.uint8))
        rows.append(np.hstack(row))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"Could not write {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare quick FUEL detector candidates on sampled frames.")
    parser.add_argument("--images", default=str(REPO_ROOT / "datasets" / "california_match_frames_2026_300" / "images"))
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "fuel_detector_comparison"))
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--fuel-model", default=str(REPO_ROOT / "models" / "flying_fuel_best.pt"))
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    image_dir = Path(args.images).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    rng = random.Random(args.seed)
    sample = rng.sample(images, min(args.sample_count, len(images)))

    model = YOLO(args.fuel_model)
    sam3_ok, sam3_message = sam3_status()
    sam3_lines = ["SAM 3", sam3_message]
    if not sam3_ok:
        sam3_lines.append("Not run in this environment.")

    rows = []
    counts: list[dict[str, object]] = []
    crop_candidates: dict[str, list[CandidateCrop]] = {
        "yolo_raw": [],
        "yolo_filtered": [],
        "hsv_contour": [],
        "hough_color": [],
    }
    sample_path = out_dir / "sample_images.txt"
    sample_path.write_text("\n".join(str(path) for path in sample) + "\n", encoding="utf-8")

    for path in sample:
        image = cv2.imread(str(path))
        if image is None:
            continue
        raw = yolo_detections(model, image, args.conf)
        filtered = yolo_shape_filter(image, raw)
        hsv = hsv_contour_detections(image)
        hough = hough_circle_detections(image)
        for method, detections in [
            ("yolo_raw", raw),
            ("yolo_filtered", filtered),
            ("hsv_contour", hsv),
            ("hough_color", hough),
        ]:
            crop_candidates[method].extend(CandidateCrop(path, image, det) for det in detections)

        rows.append(
            [
                draw_detections(image, [], path.stem[:28], (255, 255, 255)),
                draw_detections(image, raw, f"YOLO raw ({len(raw)})", (255, 80, 80)),
                draw_detections(image, filtered, f"YOLO filtered ({len(filtered)})", (80, 220, 80)),
                draw_detections(image, hsv, f"HSV contour ({len(hsv)})", (40, 180, 255)),
                draw_detections(image, hough, f"Hough/color ({len(hough)})", (210, 80, 255)),
                make_text_tile(sam3_lines, (320, 180)),
            ]
        )
        counts.append(
            {
                "image": path.name,
                "yolo_raw": len(raw),
                "yolo_filtered": len(filtered),
                "hsv_contour": len(hsv),
                "hough_color": len(hough),
                "sam3": "blocked" if not sam3_ok else "available_not_run",
            }
        )

    contact_sheet(rows, out_dir / "side_by_side.jpg", (320, 180))
    for method, candidates in crop_candidates.items():
        candidate_sheet(candidates, out_dir / f"candidate_crops_{method}.jpg")
    with (out_dir / "counts.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image", "yolo_raw", "yolo_filtered", "hsv_contour", "hough_color", "sam3"],
        )
        writer.writeheader()
        writer.writerows(counts)

    summary = {
        "sample_count": len(counts),
        "yolo_raw": sum(int(row["yolo_raw"]) for row in counts),
        "yolo_filtered": sum(int(row["yolo_filtered"]) for row in counts),
        "hsv_contour": sum(int(row["hsv_contour"]) for row in counts),
        "hough_color": sum(int(row["hough_color"]) for row in counts),
        "sam3": sam3_message,
    }
    (out_dir / "summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / 'side_by_side.jpg'}")
    print(f"Wrote candidate crop sheets to {out_dir}")
    print(f"Wrote {out_dir / 'counts.csv'}")
    print(f"Wrote {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
