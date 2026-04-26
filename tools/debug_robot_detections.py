import argparse
import os
import sys

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perspective_tracking import (
    build_camera_views,
    load_perspective_config,
    merge_multiview_detections,
    project_detections,
)


def main():
    parser = argparse.ArgumentParser(description="Sample raw robot detector output on video frames.")
    parser.add_argument("--video", default="static/downloads/youtube_pJjdRO_7KsU.mp4")
    parser.add_argument("--model", default="models/autoscouter_ventura_best_2.pt")
    parser.add_argument("--seconds", default="0,2,5,10,15,20,30,45,60,90")
    parser.add_argument("--imgsz", default="320,640,960")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--perspective-config", default="data/perspective_config.json")
    args = parser.parse_args()

    model = YOLO(args.model)
    camera_views = build_camera_views(load_perspective_config(args.perspective_config))
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    seconds = [float(value.strip()) for value in args.seconds.split(",") if value.strip()]
    image_sizes = [int(value.strip()) for value in args.imgsz.split(",") if value.strip()]

    print(f"video={args.video} size={width}x{height} fps={fps:.3f} frames={frames}")
    print(f"model={args.model} names={model.names}")

    for second in seconds:
        frame_number = int(second * fps)
        if frames and frame_number >= frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            print(f"sec={second:g} no frame")
            continue

        for image_size in image_sizes:
            result = model(frame, verbose=False, imgsz=image_size, conf=args.conf)[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                print(f"sec={second:>5g} imgsz={image_size:<4} count=0")
                continue

            confidences = boxes.conf.cpu().numpy().tolist()
            classes = boxes.cls.cpu().numpy().astype(int).tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = list(range(len(confidences)))
            projected = project_detections(
                xyxy,
                track_ids,
                confidences,
                classes,
                model.names,
                camera_views,
            )
            merged = merge_multiview_detections(projected)
            summary = [
                (model.names[class_id], round(confidence, 3))
                for class_id, confidence in zip(classes[:12], confidences[:12])
            ]
            print(
                f"sec={second:>5g} imgsz={image_size:<4} "
                f"raw={len(confidences):<3} projected={len(projected):<3} "
                f"merged={len(merged):<3} top={summary}"
            )

    cap.release()


if __name__ == "__main__":
    main()
