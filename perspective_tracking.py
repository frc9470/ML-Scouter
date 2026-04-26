import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


FIELD_WIDTH = 1379
FIELD_HEIGHT = 650
FIELD_WIDTH_INCHES = 651.222500
FIELD_SCALE = FIELD_WIDTH / FIELD_WIDTH_INCHES


def _field_inches(points):
    return [[round(x * FIELD_SCALE), round(y * FIELD_SCALE)] for x, y in points]

DEFAULT_PERSPECTIVE_CONFIG = {
    "field_size": [FIELD_WIDTH, FIELD_HEIGHT],
    "views": {
        "full": {
            "label": "Full top",
            "enabled": True,
            "source_points": [[315, 335], [1520, 317], [1586, 445], [308, 458]],
            "field_points": _field_inches(
                [[0, 0], [651.222500, 0], [651.222500, 128.043950], [0, 128.043950]]
            ),
            "point_labels": ["top-left field corner", "top-right field corner", "lower-right tower-base corner", "lower-left tower-base corner"],
        },
        "left": {
            "label": "Left tower",
            "enabled": True,
            "source_points": [[341, 802], [749, 796], [735, 952], [353, 958]],
            "field_points": _field_inches(
                [[0, 0], [158.611250, 0], [158.611250, 182.218750], [45.361558, 189.850287]]
            ),
            "point_labels": ["top-left landmark", "top-right landmark", "bottom-right tower-base landmark", "bottom-left tower-base landmark"],
        },
        "right": {
            "label": "Right tower",
            "enabled": True,
            "source_points": [[1223, 800], [1637, 802], [1619, 963], [1237, 964]],
            "field_points": _field_inches(
                [[651.222500, 0], [492.611250, 0], [492.611250, 182.218750], [605.860942, 189.850287]]
            ),
            "point_labels": ["top-right field landmark", "top-left mirrored landmark", "bottom-left tower-base landmark", "bottom-right tower-base landmark"],
        },
    },
}

VIEW_COLORS = {
    "full": (220, 180, 80),
    "left": (80, 190, 250),
    "right": (250, 140, 80),
    "fused": (80, 240, 120),
}

ALLIANCE_COLORS = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "unknown": (185, 185, 185),
}


def clone_default_config():
    return json.loads(json.dumps(DEFAULT_PERSPECTIVE_CONFIG))


def _coerce_points(points):
    if not isinstance(points, list) or len(points) != 4:
        return None
    coerced = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            coerced.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return None
    return coerced


def normalize_perspective_config(config):
    normalized = clone_default_config()
    if not isinstance(config, dict):
        return normalized

    field_size = config.get("field_size")
    if isinstance(field_size, list) and len(field_size) == 2:
        try:
            width = max(1, int(field_size[0]))
            height = max(1, int(field_size[1]))
            normalized["field_size"] = [width, height]
        except (TypeError, ValueError):
            pass

    incoming_views = config.get("views", {})
    if not isinstance(incoming_views, dict):
        return normalized

    for view_name, default_view in normalized["views"].items():
        incoming_view = incoming_views.get(view_name, {})
        if not isinstance(incoming_view, dict):
            continue

        normalized["views"][view_name]["enabled"] = bool(
            incoming_view.get("enabled", default_view["enabled"])
        )
        for key in ("source_points", "field_points"):
            points = _coerce_points(incoming_view.get(key))
            if points is not None:
                normalized["views"][view_name][key] = points
        point_labels = incoming_view.get("point_labels")
        if isinstance(point_labels, list) and len(point_labels) == 4:
            normalized["views"][view_name]["point_labels"] = [str(label) for label in point_labels]

    return normalized


def load_perspective_config(path):
    if not os.path.exists(path):
        return clone_default_config()
    with open(path, "r", encoding="utf-8") as config_file:
        return normalize_perspective_config(json.load(config_file))


def save_perspective_config(path, config):
    normalized = normalize_perspective_config(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as config_file:
        json.dump(normalized, config_file, indent=2)
    return normalized


@dataclass
class CameraView:
    name: str
    label: str
    source_points: np.ndarray
    field_points: np.ndarray
    matrix: np.ndarray


@dataclass
class FieldDetection:
    source_id: int
    bbox: Tuple[int, int, int, int]
    image_center: Tuple[int, int]
    field_point: Tuple[int, int]
    color: str
    confidence: float
    view: str
    path_id: Optional[int] = None


@dataclass
class RobotPath:
    path_id: int
    init_frame: int
    last_frame: int
    init_point: Tuple[int, int]
    last_point: Tuple[int, int]
    color: str
    points: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    def add(self, frame_number, point):
        self.points[frame_number] = point
        self.last_point = point
        self.last_frame = frame_number

    def to_dict(self):
        return {
            "id": self.path_id,
            "init_frame": self.init_frame,
            "last_frame": self.last_frame,
            "color": self.color,
            "points": {str(frame): point for frame, point in self.points.items()},
        }


def build_camera_views(config):
    config = normalize_perspective_config(config)
    views = []
    for name, view in config["views"].items():
        if not view.get("enabled", True):
            continue
        source_points = np.array(view["source_points"], dtype="float32")
        field_points = np.array(view["field_points"], dtype="float32")
        matrix = cv2.getPerspectiveTransform(source_points, field_points)
        views.append(
            CameraView(
                name=name,
                label=view.get("label", name),
                source_points=source_points,
                field_points=field_points,
                matrix=matrix,
            )
        )
    return views


def alliance_from_class(class_id, class_name):
    name = (class_name or "").strip().lower()
    if "red" in name:
        return "red"
    if "blue" in name:
        return "blue"
    return "unknown"


def _transform_point(point, matrix):
    projected = cv2.perspectiveTransform(np.array([[point]], dtype="float32"), matrix)[0][0]
    return (int(round(projected[0])), int(round(projected[1])))


def _point_in_source(point, view):
    return cv2.pointPolygonTest(view.source_points, point, False) >= 0


def project_detections(
    boxes: Iterable[Iterable[float]],
    track_ids: Iterable[int],
    confidences: Iterable[float],
    class_ids: Iterable[int],
    class_names: Dict[int, str],
    camera_views: List[CameraView],
):
    detections = []
    if not camera_views:
        return detections

    for box, track_id, confidence, class_id in zip(boxes, track_ids, confidences, class_ids):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        center = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
        matching_views = [view for view in camera_views if _point_in_source(center, view)]
        if not matching_views:
            continue

        color = alliance_from_class(int(class_id), class_names.get(int(class_id), ""))
        for view in matching_views:
            detections.append(
                FieldDetection(
                    source_id=int(track_id),
                    bbox=(x1, y1, x2, y2),
                    image_center=center,
                    field_point=_transform_point(center, view.matrix),
                    color=color,
                    confidence=float(confidence),
                    view=view.name,
                )
            )
    return detections


def _same_robot_candidate(first, second):
    if first.color != second.color:
        return False
    if first.color == "unknown" and first.view != second.view:
        return True
    return True


def merge_multiview_detections(detections, same_view_distance=40, group_distance=90):
    remaining = list(detections)
    index = 0
    while index < len(remaining) - 1:
        current = remaining[index]
        removed_current = False
        compare_index = index + 1
        while compare_index < len(remaining):
            other = remaining[compare_index]
            if (
                current.view == other.view
                and current.color == other.color
                and math.dist(current.field_point, other.field_point) < same_view_distance
            ):
                if current.confidence < other.confidence:
                    remaining.pop(index)
                    removed_current = True
                    break
                remaining.pop(compare_index)
            else:
                compare_index += 1
        if not removed_current:
            index += 1

    fused = []
    while len(remaining) >= 2:
        best_pair = None
        best_distance = group_distance
        for first_index, first in enumerate(remaining):
            for second in remaining[first_index + 1 :]:
                if first.view == second.view:
                    continue
                if not _same_robot_candidate(first, second):
                    continue
                distance = math.dist(first.field_point, second.field_point)
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (first, second)

        if best_pair is None:
            break

        first, second = best_pair
        stronger = first if first.confidence >= second.confidence else second
        midpoint = (
            int(round((first.field_point[0] + second.field_point[0]) / 2)),
            int(round((first.field_point[1] + second.field_point[1]) / 2)),
        )
        fused_detection = FieldDetection(
            source_id=first.source_id * 1000 + second.source_id,
            bbox=stronger.bbox,
            image_center=stronger.image_center,
            field_point=midpoint,
            color=first.color if first.color != "unknown" else second.color,
            confidence=first.confidence + second.confidence,
            view="fused",
        )
        fused.append(fused_detection)
        remaining = [
            detection
            for detection in remaining
            if detection not in best_pair
            and not (
                detection.color == fused_detection.color
                and detection.view != "fused"
                and math.dist(detection.field_point, midpoint) < best_distance + 1
            )
        ]

    return fused + remaining


class FieldPathTracker:
    def __init__(self, max_match_distance=150, max_missed_frames=12):
        self.max_match_distance = max_match_distance
        self.max_missed_frames = max_missed_frames
        self.active_paths = []
        self.archived_paths = []
        self.next_path_id = 1

    def update(self, frame_number, detections):
        unmatched = list(detections)
        distances = []
        for detection in unmatched:
            for path in self.active_paths:
                if detection.color != "unknown" and path.color != "unknown" and detection.color != path.color:
                    continue
                distance = math.dist(path.last_point, detection.field_point)
                if distance < self.max_match_distance:
                    distances.append((distance, detection, path))

        for _distance, detection, path in sorted(distances, key=lambda item: item[0]):
            if detection not in unmatched:
                continue
            if path not in self.active_paths:
                continue
            if frame_number in path.points:
                continue
            path.add(frame_number, detection.field_point)
            detection.path_id = path.path_id
            unmatched.remove(detection)

        for path in self.active_paths[:]:
            max_gap = min(self.max_missed_frames, len(path.points) + 1)
            if frame_number - path.last_frame > max_gap:
                self.archived_paths.append(path)
                self.active_paths.remove(path)

        for detection in unmatched:
            path = RobotPath(
                path_id=self.next_path_id,
                init_frame=frame_number,
                last_frame=frame_number,
                init_point=detection.field_point,
                last_point=detection.field_point,
                color=detection.color,
                points={frame_number: detection.field_point},
            )
            self.next_path_id += 1
            self.active_paths.append(path)
            detection.path_id = path.path_id

        return detections

    def all_paths(self):
        return self.archived_paths + self.active_paths


def make_field_canvas(config):
    config = normalize_perspective_config(config)
    width, height = config["field_size"]
    canvas = np.full((height, width, 3), (30, 35, 38), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (90, 95, 98), 2)
    cv2.line(canvas, (width // 2, 0), (width // 2, height), (65, 70, 74), 1)
    return canvas


def draw_field_overlay(canvas, paths, detections):
    for path in paths:
        color = ALLIANCE_COLORS.get(path.color, ALLIANCE_COLORS["unknown"])
        frames = sorted(path.points)
        for first_frame, second_frame in zip(frames, frames[1:]):
            cv2.line(canvas, path.points[first_frame], path.points[second_frame], color, 2)
        cv2.circle(canvas, path.last_point, 7, color, -1)
        cv2.putText(
            canvas,
            str(path.path_id),
            (path.last_point[0] + 8, path.last_point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
        )

    for detection in detections:
        color = VIEW_COLORS.get(detection.view, (120, 120, 120))
        cv2.circle(canvas, detection.field_point, 11, color, 2)


def paste_field_overlay(frame, field_canvas, width=360):
    if field_canvas is None or field_canvas.size == 0:
        return frame
    height, field_width = field_canvas.shape[:2]
    scale = width / field_width
    overlay = cv2.resize(field_canvas, (width, int(height * scale)), interpolation=cv2.INTER_AREA)
    margin = 16
    y1 = margin
    y2 = min(frame.shape[0], margin + overlay.shape[0])
    x2 = frame.shape[1] - margin
    x1 = max(0, x2 - overlay.shape[1])
    overlay = overlay[: y2 - y1, : x2 - x1]
    if overlay.size == 0:
        return frame
    roi = frame[y1:y2, x1:x2]
    blended = cv2.addWeighted(roi, 0.35, overlay, 0.65, 0)
    frame[y1:y2, x1:x2] = blended
    cv2.rectangle(frame, (x1, y1), (x2, y2), (210, 210, 210), 1)
    return frame
