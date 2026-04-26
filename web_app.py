"""
web_app.py

An all-in-one monolith file containing most of the code to run a server
analyzing and serving robot-specific shot counting metrics over a web
application.

Usage:
`conda activate frc-ai-scouter`
`python web_app.py --fuel-model path/to/model.pt \\
    --robot-model path/to/model.pt

The video to be analyzed is automatically selected in the web application
interface.
"""

import os
import cv2
import base64
import json
import re
import threading
import numpy as np
import yt_dlp
from flask import Flask, render_template, request, jsonify, Response
from collections import defaultdict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from ultralytics import YOLO
from perspective_tracking import (
    FieldPathTracker,
    build_camera_views,
    draw_field_overlay,
    load_perspective_config,
    make_field_canvas,
    merge_multiview_detections,
    paste_field_overlay,
    project_detections,
    save_perspective_config,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def project_path(*parts):
    return os.path.join(BASE_DIR, *parts)

app = Flask(__name__)
os.makedirs(project_path('static', 'crops'), exist_ok=True)
os.makedirs(project_path('static', 'downloads'), exist_ok=True)
os.makedirs(project_path('data'), exist_ok=True)

TBA_BASE_URL = "https://www.thebluealliance.com/api/v3"
PERSPECTIVE_CONFIG_PATH = project_path("data", "perspective_config.json")
FUEL_INFERENCE_SIZE = 1920
ROBOT_INFERENCE_SIZE = 640
YOLO_DEBUG_SIZE = 1920
TRACKER_CONFIGS = {
    "botsort": {"label": "BoT-SORT", "config": "botsort.yaml"},
    "bytetrack": {"label": "ByteTrack", "config": "bytetrack.yaml"},
}
COMPARE_TRACKER_MODE = "compare"

def write_local_env_value(key, value, path=None):
    path = path or project_path(".env.local")
    lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as env_file:
            lines = env_file.readlines()

    key_prefix = f"{key}="
    replacement = f"{key}={value}\n"
    for index, line in enumerate(lines):
        if line.strip().startswith(key_prefix):
            lines[index] = replacement
            break
    else:
        lines.append(replacement)

    with open(path, "w", encoding="utf-8") as env_file:
        env_file.writelines(lines)

def load_local_env(path=None):
    path = path or project_path(".env.local")
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

load_local_env()

# --- GLOBAL STATE ---
class State:
    first_frame = None
    roi_poly = None
    is_processing = False
    is_finished = False
    progress = 0
    total_frames = 1
    current_fuel_count = 0
    robot_crops = {}
    robot_scores = defaultdict(int)
    total_scored_fuel = 0
    final_team_scores = defaultdict(int)
    match_alliances = {"red": [], "blue": []}
    video_path = project_path("static", "downloads", "youtube_pJjdRO_7KsU.mp4")
    roi_frame_seconds = 0.0
    process_seconds = 20.0
    preview_frame_jpeg = None
    preview_lock = threading.Lock()
    model_status = {}
    perspective_config = load_perspective_config(PERSPECTIVE_CONFIG_PATH)
    robot_paths = []
    robot_path_count = 0
    robot_alliances = {}
    ocr_assignments = {}
    ocr_status = {"enabled": False, "message": "OCR has not run yet."}
    tracker_mode = "botsort"
    analysis_runs = {}
    active_analysis_run = None
    processing_tracker = None
    processing_pass_index = 0
    processing_pass_total = 1
    processing_status = ""

import argparse

parser = argparse.ArgumentParser(description="ML-based shot counter for FRC REBUILT")
parser.add_argument("--model", default=None, help="Path to the unified YOLO model")
parser.add_argument("--fuel-model", default=None, help="Path to the fuel YOLO model")
parser.add_argument("--robot-model", default=None, help="Path to the robot YOLO model")
parser.add_argument("--video", default=None, help="Path to the match video")

args, _unknown_args = parser.parse_known_args()

if args.video:
    State.video_path = args.video

def is_youtube_url(url):
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return parsed.scheme in {"http", "https"} and (
        host == "youtu.be" or host.endswith(".youtube.com") or host == "youtube.com"
    )

def download_youtube_video(url):
    output_template = project_path("static", "downloads", "youtube_%(id)s.%(ext)s")
    options = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = ydl.prepare_filename(info)
        if not downloaded_path.endswith(".mp4"):
            downloaded_path = os.path.splitext(downloaded_path)[0] + ".mp4"

    if not os.path.exists(downloaded_path):
        raise FileNotFoundError("Downloaded video file was not created")

    return downloaded_path

def tba_get(path):
    auth_key = os.environ.get("TBA_AUTH_KEY") or os.environ.get("TBA_API_KEY")
    if not auth_key:
        raise RuntimeError("Set TBA_AUTH_KEY or TBA_API_KEY before using the TBA picker")

    request = Request(
        f"{TBA_BASE_URL}{path}",
        headers={
            "X-TBA-Auth-Key": auth_key,
            "User-Agent": "ML-Scouter/1.0",
        },
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))

def tba_is_configured():
    return bool(os.environ.get("TBA_AUTH_KEY") or os.environ.get("TBA_API_KEY"))

def tba_error_response(error):
    if isinstance(error, RuntimeError):
        return jsonify({"error": str(error)}), 503
    if isinstance(error, HTTPError):
        return jsonify({"error": f"TBA API returned HTTP {error.code}"}), error.code
    if isinstance(error, URLError):
        return jsonify({"error": f"Could not reach TBA API: {error.reason}"}), 502
    return jsonify({"error": f"Could not query TBA API: {error}"}), 500

def match_sort_key(match):
    comp_order = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}
    return (
        comp_order.get(match.get("comp_level"), 99),
        match.get("set_number") or 0,
        match.get("match_number") or 0,
    )

def format_match_label(match):
    comp_level = match.get("comp_level", "").upper()
    set_number = match.get("set_number") or 1
    match_number = match.get("match_number") or 0
    if comp_level == "QM":
        return f"Qual {match_number}"
    return f"{comp_level} {set_number}-{match_number}"

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path if os.path.exists(video_path) else 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    duration = total_frames / fps if fps and total_frames > 0 else 0
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
    }

def encode_frame_at(video_path, seconds=0):
    cap = cv2.VideoCapture(video_path if os.path.exists(video_path) else 0)
    if seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    State.first_frame = frame
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"

def update_live_preview(frame, frame_count, stride=5, max_width=960):
    if frame_count % stride != 0:
        return

    preview_frame = frame
    height, width = preview_frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        preview_frame = cv2.resize(
            preview_frame,
            (max_width, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    ok, buffer = cv2.imencode('.jpg', preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return

    with State.preview_lock:
        State.preview_frame_jpeg = buffer.tobytes()

def letterbox_frame(frame, size):
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    return canvas, scale, pad_x, pad_y

def draw_yolo_debug_frame(frame, fuel_boxes, fuel_ids, fuel_confs, robot_boxes, robot_ids, robot_confs, robot_classes, robot_names):
    debug_frame, scale, pad_x, pad_y = letterbox_frame(frame, YOLO_DEBUG_SIZE)

    def map_box(box):
        x1, y1, x2, y2 = [float(value) for value in box]
        return (
            int(round(x1 * scale + pad_x)),
            int(round(y1 * scale + pad_y)),
            int(round(x2 * scale + pad_x)),
            int(round(y2 * scale + pad_y)),
        )

    for box, fuel_id, confidence in zip(fuel_boxes, fuel_ids, fuel_confs):
        x1, y1, x2, y2 = map_box(box)
        color = (0, 255, 255)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
        # cv2.putText(
        #     debug_frame,
        #     f"fuel {int(fuel_id)} {confidence:.2f}",
        #     (x1, max(16, y1 - 6)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.45,
        #     color,
        #     1,
        # )

    for box, robot_id, confidence, class_id in zip(robot_boxes, robot_ids, robot_confs, robot_classes):
        class_name = robot_names.get(int(class_id), "robot")
        color = (128, 0, 128)
        if class_name == "red":
            color = (0, 0, 255)
        elif class_name == "blue":
            color = (255, 0, 0)
        x1, y1, x2, y2 = map_box(box)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            debug_frame,
            f"{class_name} {int(robot_id)} {confidence:.2f}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    cv2.putText(
        debug_frame,
        f"YOLO debug {YOLO_DEBUG_SIZE}x{YOLO_DEBUG_SIZE} | fuel {len(fuel_boxes)} | robots {len(robot_boxes)}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return debug_frame

def projection_video_size(config):
    width, height = config.get("field_size", [1379, 650])
    width = int(width)
    height = int(height)
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    return width, height

def draw_projection_debug_frame(field_canvas, frame_count, raw_detections, merged_detections, paths, output_size):
    debug_frame = field_canvas.copy()
    view_colors = {
        "full": (220, 180, 80),
        "left": (80, 190, 250),
        "right": (250, 140, 80),
        "fused": (80, 240, 120),
    }

    for detection in raw_detections:
        color = view_colors.get(detection.view, (170, 170, 170))
        cv2.circle(debug_frame, detection.field_point, 16, color, 1)
        cv2.putText(
            debug_frame,
            detection.view,
            (detection.field_point[0] + 10, detection.field_point[1] + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
        )

    draw_field_overlay(debug_frame, paths, merged_detections)
    cv2.putText(
        debug_frame,
        f"Robot projection | frame {frame_count} | raw projected {len(raw_detections)} | merged {len(merged_detections)} | paths {len(paths)}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        2,
    )

    output_width, output_height = output_size
    height, width = debug_frame.shape[:2]
    if (width, height) == output_size:
        return debug_frame

    padded = np.full((output_height, output_width, 3), (30, 35, 38), dtype=np.uint8)
    padded[:height, :width] = debug_frame
    return padded

_ocr_reader = None
_ocr_error = None

def get_ocr_reader():
    global _ocr_reader, _ocr_error
    if _ocr_reader is not None:
        return _ocr_reader
    if _ocr_error is not None:
        return None

    try:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], recog_network='english_g2', user_network_directory=None)
    except Exception as e:
        _ocr_error = str(e)
        return None
    return _ocr_reader

def preprocess_robot_crop_for_ocr(crop):
    if crop is None or crop.size == 0:
        return None
    searchbox = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(searchbox, (3, 3), 0)
    searchbox = cv2.addWeighted(searchbox, 2.0, gaussian, -1.0, 0)
    return cv2.resize(searchbox, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

def normalize_team_list(values):
    teams = []
    for value in values or []:
        digits = re.sub(r"\D", "", str(value))
        if digits:
            teams.append(digits)
    return teams

def fuzzy_match_team(text, options):
    if not text or not options:
        return None, 0
    digits = re.sub(r"\D", "", str(text))
    if not digits:
        return None, 0
    if digits in options:
        return digits, 100

    try:
        from rapidfuzz import fuzz, process
        match = process.extractOne(digits, options, scorer=fuzz.ratio)
        if match:
            return match[0], int(match[1])
    except Exception:
        pass

    best_team = None
    best_score = 0
    for team in options:
        common = len(set(digits) & set(team))
        score = int((common / max(len(digits), len(team))) * 100)
        if score > best_score:
            best_team = team
            best_score = score
    return best_team, best_score

def run_robot_ocr_assignment(robot_crops, robot_alliances, match_alliances):
    reader = get_ocr_reader()
    if reader is None:
        reason = _ocr_error or "easyocr is not available"
        return {}, {
            "enabled": False,
            "message": f"OCR unavailable: {reason}. Install easyocr and rapidfuzz, then rerun analysis.",
        }

    team_options = {
        "red": normalize_team_list(match_alliances.get("red")),
        "blue": normalize_team_list(match_alliances.get("blue")),
    }
    assignments = {}

    for robot_id, crop in robot_crops.items():
        processed = preprocess_robot_crop_for_ocr(crop)
        if processed is None:
            continue

        try:
            results = reader.readtext(processed, detail=1, allowlist="0123456789")
        except Exception as e:
            assignments[str(robot_id)] = {
                "team": None,
                "text": "",
                "score": 0,
                "alliance": robot_alliances.get(robot_id, "unknown"),
                "error": str(e),
            }
            continue

        texts = []
        confidences = []
        for result in results:
            text = str(result[1]) if len(result) > 1 else ""
            digits = re.sub(r"\D", "", text)
            if not digits:
                continue
            texts.append(digits)
            confidences.append(float(result[2]) if len(result) > 2 else 0.0)

        candidate_text = max(texts, key=len) if texts else ""
        alliance = robot_alliances.get(robot_id, "unknown")
        options = team_options.get(alliance, []) if alliance in team_options else []
        if not options:
            options = sorted(set(team_options["red"] + team_options["blue"]))
        matched_team, match_score = fuzzy_match_team(candidate_text, options)
        accepted = bool(matched_team and match_score >= 70)

        assignments[str(robot_id)] = {
            "team": int(matched_team) if accepted else None,
            "candidate_team": int(matched_team) if matched_team else None,
            "text": candidate_text,
            "score": match_score,
            "ocr_confidence": max(confidences) if confidences else 0,
            "alliance": alliance,
            "accepted": accepted,
        }

    if not team_options["red"] and not team_options["blue"]:
        message = "OCR ran, but no TBA alliance team list is available for fuzzy assignment."
    else:
        accepted_count = sum(1 for assignment in assignments.values() if assignment.get("accepted"))
        message = f"OCR ran for {len(assignments)} robot crops; {accepted_count} team suggestions accepted."

    return assignments, {"enabled": True, "message": message}

# Initialize Models
unified_model = None
fuel_model = None
robot_model = None
MODEL_RUNTIME = {}

FUEL_CLASS_ALIASES = {"fuel", "ball", "flying_fuel", "flying fuel", "gamepiece", "game piece"}
ROBOT_CLASS_ALIASES = {"robot", "robots", "bot"}
COCO_CLASS_IDS = {"robot": 0, "fuel": 32}

def model_names(model):
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {index: str(name) for index, name in enumerate(names)}

def normalized_model_names(model):
    return {class_id: name.strip().lower() for class_id, name in model_names(model).items()}

def is_coco_model(model):
    names = set(normalized_model_names(model).values())
    return "person" in names and "sports ball" in names

def class_filter_for(model, target):
    if is_coco_model(model):
        return [COCO_CLASS_IDS[target]]
    return None

def class_id_for(model, target):
    if is_coco_model(model):
        return COCO_CLASS_IDS[target]

    aliases = FUEL_CLASS_ALIASES if target == "fuel" else ROBOT_CLASS_ALIASES
    for class_id, name in normalized_model_names(model).items():
        if name in aliases:
            return class_id
    return None

def class_matches_target(model, target, class_id):
    if is_coco_model(model):
        return int(class_id) == COCO_CLASS_IDS[target]

    name = normalized_model_names(model).get(int(class_id), "")
    if target == "robot":
        return (
            name in ROBOT_CLASS_ALIASES
            or "robot" in name
            or "blue" in name
            or "red" in name
        )
    return name in FUEL_CLASS_ALIASES or "fuel" in name or "ball" in name

def resolve_existing_path(path):
    if not path:
        return None
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(project_path(path))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None

def load_yolo(path, label):
    model = YOLO(path)
    names = ", ".join(f"{class_id}:{name}" for class_id, name in model_names(model).items())
    print(f"Loaded {label} model: {path} [{names}]")
    return model

def valid_tracker_keys():
    return tuple(TRACKER_CONFIGS.keys())

def normalize_tracker_mode(value):
    value = str(value or "").strip().lower()
    if value == COMPARE_TRACKER_MODE:
        return COMPARE_TRACKER_MODE
    if value in TRACKER_CONFIGS:
        return value
    return "botsort"

def tracker_runs_for_mode(mode):
    mode = normalize_tracker_mode(mode)
    if mode == COMPARE_TRACKER_MODE:
        return ["botsort", "bytetrack"]
    return [mode]

def tracker_meta(key):
    return TRACKER_CONFIGS[key]

def tracker_label(key):
    return tracker_meta(key)["label"]

def tracker_config_name(key):
    return tracker_meta(key)["config"]

def tracker_output_paths(key):
    return {
        "annotated": project_path("static", f"output_{key}.mp4"),
        "yolo_debug": project_path("static", f"yolo_debug_{key}.mp4"),
        "projection": project_path("static", f"robot_projection_{key}.mp4"),
    }

def tracker_video_urls(key):
    return {
        "annotated": f"/static/output_{key}.mp4",
        "yolo_debug": f"/static/yolo_debug_{key}.mp4",
        "projection": f"/static/robot_projection_{key}.mp4",
    }

def clear_preview_frame():
    with State.preview_lock:
        State.preview_frame_jpeg = None

def reset_analysis_state():
    State.is_finished = False
    State.progress = 0
    State.current_fuel_count = 0
    State.robot_crops = {}
    State.robot_scores = defaultdict(int)
    State.total_scored_fuel = 0
    State.final_team_scores = defaultdict(int)
    State.robot_paths = []
    State.robot_path_count = 0
    State.robot_alliances = {}
    State.ocr_assignments = {}
    State.ocr_status = {"enabled": False, "message": "OCR has not run yet."}
    State.analysis_runs = {}
    State.active_analysis_run = None
    State.processing_tracker = None
    State.processing_pass_index = 0
    State.processing_pass_total = 1
    State.processing_status = ""
    clear_preview_frame()

def activate_analysis_run(key):
    snapshot = State.analysis_runs.get(key)
    if not snapshot:
        return False
    State.active_analysis_run = key
    State.robot_crops = snapshot["robot_crops"]
    State.robot_scores = defaultdict(int, snapshot["robot_scores"])
    State.total_scored_fuel = snapshot["total_scored_fuel"]
    State.robot_paths = snapshot["robot_paths"]
    State.robot_path_count = snapshot["robot_path_count"]
    State.robot_alliances = snapshot["robot_alliances"]
    State.ocr_assignments = snapshot["ocr_assignments"]
    State.ocr_status = snapshot["ocr_status"]
    return True

def analysis_run_summaries():
    runs = []
    for key in tracker_runs_for_mode(State.tracker_mode):
        snapshot = State.analysis_runs.get(key)
        if not snapshot:
            continue
        summary = dict(snapshot["summary"])
        summary["active"] = key == State.active_analysis_run
        summary["video_urls"] = snapshot["video_urls"]
        runs.append(summary)
    return runs

# Resolve model paths: explicit flags > bundled AutoScouter robot model > uploaded models/ dir > unified > YOLO11n.
_fuel_path = resolve_existing_path(args.fuel_model) or resolve_existing_path(
    project_path("models", "flying_fuel_best.pt")
)
_robot_path = (
    resolve_existing_path(args.robot_model)
    or resolve_existing_path(project_path("models", "autoscouter_ventura_best_2.pt"))
    or resolve_existing_path(project_path("models", "robot_best.pt"))
)

if _fuel_path and _robot_path:
    try:
        fuel_model = load_yolo(_fuel_path, "fuel")
        robot_model = load_yolo(_robot_path, "robot")
        MODEL_RUNTIME = {
            "mode": "separate",
            "fuel_path": _fuel_path,
            "robot_path": _robot_path,
        }
        State.model_status = {
            "mode": "separate",
            "fuel_model": _fuel_path,
            "robot_model": _robot_path,
            "fuel_names": model_names(fuel_model),
            "robot_names": model_names(robot_model),
        }
    except Exception as e:
        print(f"ERROR: Could not load uploaded fuel/robot models: {e}")
        fallback_path = resolve_existing_path("yolo11n.pt") or "yolo11n.pt"
        unified_model = load_yolo(fallback_path, "COCO fallback")
        MODEL_RUNTIME = {
            "mode": "unified",
            "model_path": fallback_path,
        }
        State.model_status = {"mode": "coco_fallback", "error": str(e), "model": fallback_path}
else:
    model_path = resolve_existing_path(args.model) or resolve_existing_path("unified.pt")
    try:
        if not model_path:
            raise FileNotFoundError("No uploaded separate models or unified.pt found")
        unified_model = load_yolo(model_path, "unified")
        MODEL_RUNTIME = {
            "mode": "unified",
            "model_path": model_path,
        }
        State.model_status = {
            "mode": "unified",
            "model": model_path,
            "names": model_names(unified_model),
        }
    except Exception as e:
        print(f"ERROR: Could not load custom model. Defaulting to YOLO11n: {e}")
        fallback_path = resolve_existing_path("yolo11n.pt") or "yolo11n.pt"
        unified_model = load_yolo(fallback_path, "COCO fallback")
        MODEL_RUNTIME = {
            "mode": "unified",
            "model_path": fallback_path,
        }
        State.model_status = {"mode": "coco_fallback", "error": str(e), "model": fallback_path}

def load_models_for_analysis():
    if MODEL_RUNTIME.get("mode") == "separate":
        fresh_fuel_model = load_yolo(MODEL_RUNTIME["fuel_path"], "fuel")
        fresh_robot_model = load_yolo(MODEL_RUNTIME["robot_path"], "robot")
        return {
            "unified_model": None,
            "fuel_model": fresh_fuel_model,
            "robot_model": fresh_robot_model,
            "robot_class_names": model_names(fresh_robot_model),
        }

    fresh_unified_model = load_yolo(MODEL_RUNTIME["model_path"], "unified")
    return {
        "unified_model": fresh_unified_model,
        "fuel_model": None,
        "robot_model": None,
        "robot_class_names": model_names(fresh_unified_model),
    }

# --- KINEMATIC HELPERS ---
def is_airborne(history, min_frames=3, velocity_threshold=2.0):
    if len(history) < min_frames:
        return False
    current_y = history[-1]['y']
    older_y = history[-min_frames]['y']
    if (older_y - current_y) > velocity_threshold:
        return True
    return False

def calculate_trajectory_and_attribute(fuel_id, fuel_history, robot_histories):
    if len(fuel_history) < 3:
        return None
    frames = [pt['frame'] for pt in fuel_history]
    X = np.array([pt['x'] for pt in fuel_history])
    Y = np.array([pt['y'] for pt in fuel_history])
    launch_frame = frames[0]
    launch_x = X[0]
    
    try:
        with np.errstate(all='ignore'):
            coeffs = np.polyfit(X, Y, 2)
            origin_y = np.polyval(coeffs, launch_x)
    except Exception:
        origin_y = Y[0]
    
    origin_point = (launch_x, origin_y)
    
    closest_robot_id = None
    min_distance = float('inf')
    for robot_id, rob_history in robot_histories.items():
        rob_state_at_launch = next((state for state in rob_history if state['frame'] == launch_frame), None)
        if rob_state_at_launch:
            rx1, ry1, rx2, ry2 = rob_state_at_launch['bbox']
            top_center_x = (rx1 + rx2) / 2.0
            top_center_y = ry1
            dist = np.sqrt((origin_point[0] - top_center_x)**2 + (origin_point[1] - top_center_y)**2)
            if dist < min_distance:
                min_distance = dist
                closest_robot_id = robot_id
    return closest_robot_id

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/first_frame')
def api_first_frame():
    image = encode_frame_at(State.video_path, State.roi_frame_seconds)
    if not image:
        return jsonify({"error": "Could not read video"}), 500
    return jsonify({
        "image": image,
        "video_path": State.video_path,
        "seconds": State.roi_frame_seconds,
        "video": get_video_info(State.video_path),
    })

@app.route('/api/frame_at', methods=['POST'])
def api_frame_at():
    if State.is_processing:
        return jsonify({"error": "Cannot change ROI frame while processing"}), 409

    data = request.json or {}
    try:
        seconds = max(0.0, float(data.get('seconds', 0)))
    except (TypeError, ValueError):
        return jsonify({"error": "Frame time must be a number of seconds"}), 400

    image = encode_frame_at(State.video_path, seconds)
    if not image:
        return jsonify({"error": "Could not read frame at that time"}), 500

    State.roi_frame_seconds = seconds
    State.roi_poly = None
    return jsonify({
        "success": True,
        "image": image,
        "seconds": State.roi_frame_seconds,
        "video": get_video_info(State.video_path),
    })

@app.route('/api/set_video_source', methods=['POST'])
def api_set_video_source():
    if State.is_processing:
        return jsonify({"error": "Cannot change video while processing"}), 409

    data = request.json or {}
    url = data.get('youtube_url', '').strip()
    if not url:
        return jsonify({"error": "YouTube URL is required"}), 400
    if not is_youtube_url(url):
        return jsonify({"error": "Please enter a valid YouTube URL"}), 400

    try:
        video_path = download_youtube_video(url)
        image = encode_frame_at(video_path, 0)
    except Exception as e:
        return jsonify({"error": f"Could not download video: {e}"}), 500

    if not image:
        return jsonify({"error": "Downloaded video could not be read"}), 500

    State.video_path = video_path
    State.roi_frame_seconds = 0.0
    State.roi_poly = None
    reset_analysis_state()
    State.match_alliances = {
        "red": normalize_team_list(data.get("red", [])),
        "blue": normalize_team_list(data.get("blue", [])),
    }

    return jsonify({
        "success": True,
        "image": image,
        "video_path": State.video_path,
        "seconds": State.roi_frame_seconds,
        "video": get_video_info(State.video_path),
    })

@app.route('/api/tba/config', methods=['GET', 'POST'])
def api_tba_config():
    if request.method == 'GET':
        return jsonify({"configured": tba_is_configured()})

    data = request.json or {}
    key = data.get("auth_key", "").strip()
    if not key:
        return jsonify({"error": "TBA API key is required"}), 400

    try:
        write_local_env_value("TBA_AUTH_KEY", key)
        os.environ["TBA_AUTH_KEY"] = key
    except Exception as e:
        return jsonify({"error": f"Could not save TBA API key: {e}"}), 500

    return jsonify({"success": True, "configured": True})

@app.route('/api/tba/events')
def api_tba_events():
    try:
        year = int(request.args.get('year', ''))
    except ValueError:
        return jsonify({"error": "Year must be a number"}), 400

    query = request.args.get('q', '').strip().lower()
    try:
        events = tba_get(f"/events/{year}/simple")
    except Exception as e:
        return tba_error_response(e)

    if query:
        events = [
            event for event in events
            if query in event.get("name", "").lower()
            or query in event.get("key", "").lower()
            or query in event.get("city", "").lower()
            or query in event.get("state_prov", "").lower()
            or query in event.get("country", "").lower()
        ]

    events = sorted(events, key=lambda event: (event.get("start_date") or "", event.get("name") or ""))
    return jsonify({
        "events": [
            {
                "key": event.get("key"),
                "name": event.get("name"),
                "city": event.get("city"),
                "state_prov": event.get("state_prov"),
                "country": event.get("country"),
                "start_date": event.get("start_date"),
                "end_date": event.get("end_date"),
            }
            for event in events[:75]
        ]
    })

@app.route('/api/tba/event/<event_key>/matches')
def api_tba_event_matches(event_key):
    try:
        matches = tba_get(f"/event/{event_key}/matches")
    except Exception as e:
        return tba_error_response(e)

    match_options = []
    for match in sorted(matches, key=match_sort_key):
        youtube_videos = [
            video for video in match.get("videos", [])
            if video.get("type") == "youtube" and video.get("key")
        ]
        if not youtube_videos:
            continue

        youtube_key = youtube_videos[0]["key"]
        red = [team.replace("frc", "") for team in match.get("alliances", {}).get("red", {}).get("team_keys", [])]
        blue = [team.replace("frc", "") for team in match.get("alliances", {}).get("blue", {}).get("team_keys", [])]
        match_options.append({
            "key": match.get("key"),
            "label": format_match_label(match),
            "youtube_key": youtube_key,
            "youtube_url": f"https://www.youtube.com/watch?v={youtube_key}",
            "red": red,
            "blue": blue,
            "time": match.get("time") or match.get("actual_time"),
        })

    return jsonify({"matches": match_options})

@app.route('/api/set_roi', methods=['POST'])
def api_set_roi():
    data = request.json or {}
    points = data.get('points', [])
    if len(points) < 3:
        return jsonify({"error": "Need at least 3 points"}), 400

    try:
        process_seconds = float(data.get('process_seconds', State.process_seconds))
    except (TypeError, ValueError):
        return jsonify({"error": "Analyze duration must be a number of seconds"}), 400
    if process_seconds < 0:
        return jsonify({"error": "Analyze duration cannot be negative"}), 400
    tracker_mode = normalize_tracker_mode(data.get("tracker_mode"))
    
    State.roi_poly = np.array(points, np.int32).reshape((-1, 1, 2))
    State.process_seconds = process_seconds
    State.tracker_mode = tracker_mode
    State.is_processing = True
    reset_analysis_state()
    State.is_processing = True
    State.processing_pass_total = len(tracker_runs_for_mode(tracker_mode))
    
    # Start background processing thread
    threading.Thread(target=process_video_task).start()
    return jsonify({"success": True})

@app.route('/api/perspective_config')
def api_perspective_config():
    return jsonify(State.perspective_config)

@app.route('/api/set_perspective_config', methods=['POST'])
def api_set_perspective_config():
    if State.is_processing:
        return jsonify({"error": "Cannot change perspective calibration while processing"}), 409

    try:
        State.perspective_config = save_perspective_config(
            PERSPECTIVE_CONFIG_PATH,
            request.json or {},
        )
    except Exception as e:
        return jsonify({"error": f"Could not save perspective calibration: {e}"}), 400

    return jsonify({"success": True, "config": State.perspective_config})

@app.route('/api/select_analysis_run', methods=['POST'])
def api_select_analysis_run():
    if State.is_processing:
        return jsonify({"error": "Cannot change analysis run while processing"}), 409

    run_key = normalize_tracker_mode((request.json or {}).get("run_key"))
    if run_key == COMPARE_TRACKER_MODE:
        return jsonify({"error": "Select a concrete tracker run"}), 400
    if not activate_analysis_run(run_key):
        return jsonify({"error": f"Analysis run '{run_key}' is not available"}), 404

    return jsonify({
        "success": True,
        "active_analysis_run": State.active_analysis_run,
        "analysis_runs": analysis_run_summaries(),
    })

def run_analysis_pass(tracker_key, progress_offset, per_pass_frame_limit):
    model_bundle = load_models_for_analysis()
    active_unified_model = model_bundle["unified_model"]
    active_fuel_model = model_bundle["fuel_model"]
    active_robot_model = model_bundle["robot_model"]
    robot_class_names = model_bundle["robot_class_names"]

    cap = cv2.VideoCapture(State.video_path if os.path.exists(State.video_path) else 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_paths = tracker_output_paths(tracker_key)
    projection_size = projection_video_size(State.perspective_config)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_paths["annotated"], fourcc, fps, (width, height))
    yolo_out = cv2.VideoWriter(output_paths["yolo_debug"], fourcc, fps, (YOLO_DEBUG_SIZE, YOLO_DEBUG_SIZE))
    projection_out = cv2.VideoWriter(output_paths["projection"], fourcc, fps, projection_size)
    
    fuel_history = defaultdict(list)
    robot_history = defaultdict(list)
    scored_fuel_ids = set()
    frame_count = 0
    total_fuel_detections = 0
    frames_with_fuel = 0
    max_live_fuel_count = 0
    camera_views = build_camera_views(State.perspective_config)
    field_tracker = FieldPathTracker()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        State.progress = progress_offset + frame_count
        display_frame = frame.copy()

        if State.roi_poly is not None:
            cv2.polylines(display_frame, [State.roi_poly], True, (255, 255, 255), 2)

        robot_boxes_list, robot_ids_list, robot_conf_list, robot_class_list = [], [], [], []
        fuel_boxes_list, fuel_ids_list, fuel_conf_list = [], [], []

        if active_unified_model:
            results = active_unified_model.track(
                frame,
                persist=True,
                verbose=False,
                imgsz=ROBOT_INFERENCE_SIZE,
                tracker=tracker_config_name(tracker_key),
            )
            
            robot_cls = class_id_for(active_unified_model, "robot")
            fuel_cls = class_id_for(active_unified_model, "fuel")

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for b, t_id, conf, c in zip(boxes, ids, confidences, classes):
                    if (robot_cls is not None and c == robot_cls) or class_matches_target(active_unified_model, "robot", c):
                        robot_boxes_list.append(b)
                        robot_ids_list.append(t_id)
                        robot_conf_list.append(conf)
                        robot_class_list.append(c)
                    elif (fuel_cls is not None and c == fuel_cls) or class_matches_target(active_unified_model, "fuel", c):
                        fuel_boxes_list.append(b)
                        fuel_ids_list.append(t_id)
                        fuel_conf_list.append(conf)
        else:
            fuel_results = active_fuel_model.track(
                frame,
                persist=True,
                verbose=False,
                imgsz=FUEL_INFERENCE_SIZE,
                classes=class_filter_for(active_fuel_model, "fuel"),
                tracker=tracker_config_name(tracker_key),
            )
            robot_results = active_robot_model.track(
                frame,
                persist=True,
                verbose=False,
                imgsz=ROBOT_INFERENCE_SIZE,
                classes=class_filter_for(active_robot_model, "robot"),
                tracker=tracker_config_name(tracker_key),
            )
            
            if robot_results[0].boxes.id is not None:
                boxes = robot_results[0].boxes.xyxy.cpu().numpy()
                ids = robot_results[0].boxes.id.cpu().numpy().astype(int)
                confidences = robot_results[0].boxes.conf.cpu().numpy()
                classes = robot_results[0].boxes.cls.cpu().numpy().astype(int)
                for b, t_id, conf, c in zip(boxes, ids, confidences, classes):
                    robot_boxes_list.append(b)
                    robot_ids_list.append(t_id)
                    robot_conf_list.append(conf)
                    robot_class_list.append(c)
                    
            if fuel_results[0].boxes.id is not None:
                boxes = fuel_results[0].boxes.xyxy.cpu().numpy()
                ids = fuel_results[0].boxes.id.cpu().numpy().astype(int)
                confidences = fuel_results[0].boxes.conf.cpu().numpy()
                for b, t_id, conf in zip(boxes, ids, confidences):
                    fuel_boxes_list.append(b)
                    fuel_ids_list.append(t_id)
                    fuel_conf_list.append(conf)

        State.current_fuel_count = len(fuel_boxes_list)
        total_fuel_detections += len(fuel_boxes_list)
        if fuel_boxes_list:
            frames_with_fuel += 1
        max_live_fuel_count = max(max_live_fuel_count, len(fuel_boxes_list))
        yolo_debug_frame = draw_yolo_debug_frame(
            frame,
            fuel_boxes_list,
            fuel_ids_list,
            fuel_conf_list,
            robot_boxes_list,
            robot_ids_list,
            robot_conf_list,
            robot_class_list,
            robot_class_names,
        )

        raw_field_detections = project_detections(
            robot_boxes_list,
            robot_ids_list,
            robot_conf_list,
            robot_class_list,
            robot_class_names,
            camera_views,
        )
        field_detections = merge_multiview_detections(raw_field_detections)
        field_detections = field_tracker.update(frame_count, field_detections)
        State.robot_path_count = len(field_tracker.all_paths())
        field_canvas = make_field_canvas(State.perspective_config)
        projection_debug_frame = draw_projection_debug_frame(
            field_canvas,
            frame_count,
            raw_field_detections,
            field_detections,
            field_tracker.all_paths(),
            projection_size,
        )

        projected_source_ids = {detection.source_id for detection in raw_field_detections}
        for box, raw_id in zip(robot_boxes_list, robot_ids_list):
            if int(raw_id) in projected_source_ids:
                continue
            rx1, ry1, rx2, ry2 = map(int, box)
            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 180, 255), 2)
            cv2.putText(
                display_frame,
                f"Raw robot {int(raw_id)}",
                (rx1, max(20, ry1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 255),
                2,
            )

        # Process Robots
        for detection in field_detections:
            r_id = detection.path_id
            rx1, ry1, rx2, ry2 = detection.bbox
            robot_history[r_id].append({
                'frame': frame_count,
                'bbox': detection.bbox,
                'field_point': detection.field_point,
                'color': detection.color,
                'view': detection.view,
            })
            
            if r_id not in State.robot_crops or frame_count % 30 == 0:
                crop = frame[max(0, ry1-10):ry2+10, max(0, rx1-10):rx2+10]
                if crop.size > 0:
                    State.robot_crops[r_id] = crop
                    State.robot_alliances[r_id] = detection.color

            box_color = (128, 0, 128)
            if detection.color == "red":
                box_color = (0, 0, 255)
            elif detection.color == "blue":
                box_color = (255, 0, 0)
            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), box_color, 3)
            cv2.putText(
                display_frame,
                f"Robot {r_id} {detection.view}",
                (rx1, max(20, ry1-10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
            )

        # Process FUEL
        for box, f_id in zip(fuel_boxes_list, fuel_ids_list):
            fx1, fy1, fx2, fy2 = map(int, box)
            cx = int((fx1 + fx2) / 2)
            cy = int((fy1 + fy2) / 2)
            fuel_history[f_id].append({'frame': frame_count, 'x': cx, 'y': cy, 'bbox': (fx1, fy1, fx2, fy2)})

            if f_id not in scored_fuel_ids and State.roi_poly is not None:
                inside = cv2.pointPolygonTest(State.roi_poly, (cx, cy), False) >= 0
                if inside:
                    scored_fuel_ids.add(f_id)
                    State.total_scored_fuel += 1
                    source_robot_id = calculate_trajectory_and_attribute(f_id, fuel_history[f_id], robot_history)
                    if source_robot_id is not None:
                        State.robot_scores[source_robot_id] += 1

            color = (0, 255, 0) if f_id in scored_fuel_ids else (0, 0, 255)
            cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), color, 2)
            if f_id in scored_fuel_ids:
                cv2.putText(display_frame, "SCORED", (fx1, fy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            pts = np.array([[pt['x'], pt['y']] for pt in fuel_history[f_id]], np.int32)
            cv2.polylines(display_frame, [pts], False, color, 1)

        draw_field_overlay(field_canvas, field_tracker.all_paths(), field_detections)
        paste_field_overlay(display_frame, field_canvas)

        cv2.putText(display_frame, f"Scored FUEL: {State.total_scored_fuel}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        update_live_preview(display_frame, frame_count)
        out.write(display_frame)
        yolo_out.write(yolo_debug_frame)
        projection_out.write(projection_debug_frame)

        if per_pass_frame_limit > 0 and frame_count >= per_pass_frame_limit:
            break

    cap.release()
    out.release()
    yolo_out.release()
    projection_out.release()

    robot_paths = [path.to_dict() for path in field_tracker.all_paths()]
    robot_path_count = len(robot_paths)
    ocr_assignments, ocr_status = run_robot_ocr_assignment(
        State.robot_crops.copy(),
        State.robot_alliances.copy(),
        State.match_alliances,
    )
    runtime_seconds = frame_count / fps if fps and frame_count > 0 else 0.0

    return {
        "key": tracker_key,
        "label": tracker_label(tracker_key),
        "robot_crops": dict(State.robot_crops),
        "robot_scores": dict(State.robot_scores),
        "total_scored_fuel": State.total_scored_fuel,
        "robot_paths": robot_paths,
        "robot_path_count": robot_path_count,
        "robot_alliances": dict(State.robot_alliances),
        "ocr_assignments": ocr_assignments,
        "ocr_status": ocr_status,
        "video_urls": tracker_video_urls(tracker_key),
        "summary": {
            "key": tracker_key,
            "label": tracker_label(tracker_key),
            "tracker_config": tracker_config_name(tracker_key),
            "frames_processed": frame_count,
            "runtime_seconds": round(runtime_seconds, 2),
            "unique_fuel_tracks": len(fuel_history),
            "fuel_detections": total_fuel_detections,
            "frames_with_fuel": frames_with_fuel,
            "avg_fuel_detections_per_frame": round(total_fuel_detections / frame_count, 3) if frame_count else 0,
            "max_live_fuel_count": max_live_fuel_count,
            "total_scored": State.total_scored_fuel,
            "robot_path_count": robot_path_count,
        },
    }

def process_video_task():
    runs = tracker_runs_for_mode(State.tracker_mode)
    cap = cv2.VideoCapture(State.video_path if os.path.exists(State.video_path) else 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    max_frames = int(State.process_seconds * fps) if State.process_seconds > 0 else source_total_frames
    if source_total_frames > 0 and max_frames > 0:
        per_pass_frame_limit = min(source_total_frames, max_frames)
    elif max_frames > 0:
        per_pass_frame_limit = max_frames
    elif source_total_frames > 0:
        per_pass_frame_limit = source_total_frames
    else:
        per_pass_frame_limit = 1

    State.total_frames = max(1, per_pass_frame_limit * len(runs))
    State.processing_pass_total = len(runs)

    try:
        for index, tracker_key in enumerate(runs, start=1):
            State.current_fuel_count = 0
            State.robot_crops = {}
            State.robot_scores = defaultdict(int)
            State.total_scored_fuel = 0
            State.robot_paths = []
            State.robot_path_count = 0
            State.robot_alliances = {}
            State.ocr_assignments = {}
            State.ocr_status = {"enabled": False, "message": "OCR has not run yet."}
            clear_preview_frame()
            State.is_processing = True
            State.processing_pass_index = index
            State.processing_pass_total = len(runs)
            State.processing_tracker = tracker_key
            State.processing_status = f"Running {tracker_label(tracker_key)} ({index}/{len(runs)})"
            snapshot = run_analysis_pass(
                tracker_key,
                progress_offset=(index - 1) * per_pass_frame_limit,
                per_pass_frame_limit=per_pass_frame_limit,
            )
            State.analysis_runs[tracker_key] = snapshot

        if runs:
            activate_analysis_run(runs[0])
        State.current_fuel_count = 0
        State.progress = State.total_frames
        State.processing_status = "Analysis complete."
        State.is_finished = True
    except Exception as error:
        State.processing_status = f"Analysis failed: {error}"
        State.ocr_status = {"enabled": False, "message": State.processing_status}
        State.is_finished = False
    finally:
        State.is_processing = False

@app.route('/api/status')
def api_status():
    return jsonify({
        "is_processing": State.is_processing,
        "is_finished": State.is_finished,
        "progress": State.progress,
        "total_frames": State.total_frames,
        "total_scored": State.total_scored_fuel,
        "current_fuel_count": State.current_fuel_count,
        "process_seconds": State.process_seconds,
        "preview_available": State.preview_frame_jpeg is not None,
        "model_status": State.model_status,
        "robot_path_count": State.robot_path_count,
        "ocr_status": State.ocr_status,
        "tracker_mode": State.tracker_mode,
        "active_analysis_run": State.active_analysis_run,
        "analysis_runs": analysis_run_summaries(),
        "processing_tracker": State.processing_tracker,
        "processing_pass_index": State.processing_pass_index,
        "processing_pass_total": State.processing_pass_total,
        "processing_status": State.processing_status,
    })

@app.route('/api/live_frame')
def api_live_frame():
    with State.preview_lock:
        frame = State.preview_frame_jpeg

    if frame is None:
        return Response(status=204)

    return Response(frame, mimetype='image/jpeg')

@app.route('/api/results')
def api_results():
    crops_b64 = {}
    for r_id, crop in State.robot_crops.items():
        _, buffer = cv2.imencode('.jpg', crop)
        crops_b64[r_id] = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "scores": State.robot_scores,
        "crops": crops_b64,
        "total_scored": State.total_scored_fuel,
        "robot_paths": State.robot_paths,
        "ocr_assignments": State.ocr_assignments,
        "ocr_status": State.ocr_status,
        "match_alliances": State.match_alliances,
        "active_analysis_run": State.active_analysis_run,
        "analysis_runs": analysis_run_summaries(),
    })

@app.route('/api/submit_attribution', methods=['POST'])
def api_submit_attribution():
    data = request.json
    team_mapping = data.get('mapping', {})
    
    final_scores = defaultdict(int)
    unattributed = State.total_scored_fuel - sum(State.robot_scores.values())
    
    for r_id_str, team_num_str in team_mapping.items():
        r_id = int(r_id_str)
        score = State.robot_scores.get(r_id, 0)
        try:
            team_num = int(team_num_str)
            final_scores[team_num] += score
        except ValueError:
            unattributed += score
            
    State.final_team_scores = final_scores
    
    return jsonify({
        "final_scores": dict(final_scores),
        "unattributed": unattributed
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)
