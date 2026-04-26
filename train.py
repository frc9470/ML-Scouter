"""
Train a YOLO model on the Roboflow export in train-test-split/.

Labels are polygon (segmentation) format; this script defaults to YOLO11n-seg.
After training, point shot_counter_ml.py at the printed best.pt and set
BALL_CLASS_ID = 0.

Dependencies:
    pip install ultralytics

Usage (from repo root):
    python train.py
    python train.py --epochs 100 --imgsz 640 --batch 8
    python train.py --dry-run
    python train.py --resume runs/ball_seg/weights/last.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = REPO_ROOT / "train-test-split"


def _count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in exts)


def build_dataset_yaml() -> Path:
    """Ultralytics expects train/val paths relative to dataset root; Roboflow's data.yaml uses broken ../ paths."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_ROOT}")

    train_img = DATASET_ROOT / "train" / "images"
    val_img = DATASET_ROOT / "valid" / "images"
    if not train_img.is_dir() or not val_img.is_dir():
        raise FileNotFoundError(
            f"Expected {train_img} and {val_img} (train and valid image folders)."
        )

    root = DATASET_ROOT.resolve()
    lines = [
        f"path: {root.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "nc: 1",
        "names: ['ball']",
        "",
    ]
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="ball_dataset_", text=True)
    os.close(fd)
    p = Path(path)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def dataset_yaml_text() -> str:
    """Inline YAML for --dry-run (no temp file)."""
    root = DATASET_ROOT.resolve()
    return "\n".join(
        [
            f"path: {root.as_posix()}",
            "train: train/images",
            "val: valid/images",
            "nc: 1",
            "names: ['ball']",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on train-test-split/")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print dataset layout and effective data.yaml, then exit (no Ultralytics import).",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Base weights when not using --resume (segment model; matches polygon labels).",
    )
    parser.add_argument(
        "--resume",
        metavar="LAST_PT",
        default=None,
        help="Path to last.pt from a previous run; loads that checkpoint and continues training.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader workers (lower if you hit shared memory errors).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="e.g. 0, cpu, or 0,1 — passed through to Ultralytics",
    )
    parser.add_argument(
        "--project",
        default=str(REPO_ROOT / "runs"),
        help="Project directory for Ultralytics runs",
    )
    parser.add_argument(
        "--name",
        default="ball_seg",
        help="Run name (Ultralytics may append 2, 3, … if the folder exists)",
    )
    args = parser.parse_args()

    train_img = DATASET_ROOT / "train" / "images"
    val_img = DATASET_ROOT / "valid" / "images"

    if args.dry_run:
        print("=== dry-run ===")
        print(f"Dataset root: {DATASET_ROOT.resolve()}")
        print(f"Train images: {train_img}  ({_count_images(train_img)} files)")
        print(f"Val images:   {val_img}  ({_count_images(val_img)} files)")
        print("Effective data.yaml:")
        print(dataset_yaml_text())
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    data_yaml = build_dataset_yaml()
    print(f"Dataset config: {data_yaml}")
    print(f"Dataset root:   {DATASET_ROOT.resolve()}")
    print(f"Train / val:    {_count_images(train_img)} / {_count_images(val_img)} images")

    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            print(f"Resume checkpoint not found: {resume_path}", file=sys.stderr)
            sys.exit(1)
        model = YOLO(str(resume_path))
        train_resume = True
        print(f"Resuming from: {resume_path}")
    else:
        model = YOLO(args.model)
        train_resume = False
        print(f"Starting from: {args.model}")

    kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=train_resume,
        device="0"
    )
    if args.device is not None:
        kwargs["device"] = args.device

    try:
        metrics = model.train(**kwargs)
    finally:
        try:
            data_yaml.unlink(missing_ok=True)
        except OSError:
            pass

    best_pt = None
    trainer = getattr(model, "trainer", None)
    if trainer is not None and getattr(trainer, "best", None) is not None:
        b = trainer.best
        if hasattr(b, "exists") and b.exists():
            best_pt = Path(b)

    if best_pt is None:
        fallback = Path(args.project) / args.name / "weights" / "best.pt"
        if fallback.is_file():
            best_pt = fallback

    print()
    if best_pt is not None and best_pt.is_file():
        print(f"Best weights: {best_pt.resolve()}")
    else:
        print(
            "Could not locate best.pt automatically; check Ultralytics output under",
            f"{args.project!r}.",
        )

    if metrics is not None:
        print(f"\nTraining metrics: {metrics}")


if __name__ == "__main__":
    main()
