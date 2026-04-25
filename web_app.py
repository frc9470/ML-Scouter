import os
import cv2
import base64
import threading
import numpy as np
import yt_dlp
from flask import Flask, render_template, request, jsonify
from collections import defaultdict
from urllib.parse import urlparse
from ultralytics import YOLO

app = Flask(__name__)
os.makedirs('static/crops', exist_ok=True)
os.makedirs('static/downloads', exist_ok=True)

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
    video_path = "match_video.mp4"
    roi_frame_seconds = 0.0

import argparse

parser = argparse.ArgumentParser(description="ML-based shot counter for FRC REBUILT")
parser.add_argument("--model", default=None, help="Path to the unified YOLO model")
parser.add_argument("--fuel-model", default=None, help="Path to the fuel YOLO model")
parser.add_argument("--robot-model", default=None, help="Path to the robot YOLO model")
parser.add_argument("--video", default="match_video.mp4", help="Path to the match video")

args, _unknown_args = parser.parse_known_args()

State.video_path = args.video

def is_youtube_url(url):
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return parsed.scheme in {"http", "https"} and (
        host == "youtu.be" or host.endswith(".youtube.com") or host == "youtube.com"
    )

def download_youtube_video(url):
    output_template = os.path.abspath("static/downloads/youtube_%(id)s.%(ext)s")
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

# Initialize Models
unified_model = None
fuel_model = None
robot_model = None

if args.fuel_model and args.robot_model:
    try:
        fuel_model = YOLO(args.fuel_model)
        robot_model = YOLO(args.robot_model)
    except Exception as e:
        print(f"ERROR: Could not load individual fuel and robot models, defaulting to YOLO11n: {e}")
        unified_model = YOLO('yolo11n.pt')
else:
    model_path = args.model if args.model else "unified.pt"
    try:
        unified_model = YOLO(model_path) 
    except Exception:
        print(f"ERROR: Could not load unified model. Defaulting to YOLO11n.")
        unified_model = YOLO('yolo11n.pt')

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
    State.is_finished = False
    State.progress = 0
    State.current_fuel_count = 0
    State.robot_crops = {}
    State.robot_scores = defaultdict(int)
    State.total_scored_fuel = 0
    State.final_team_scores = defaultdict(int)

    return jsonify({
        "success": True,
        "image": image,
        "video_path": State.video_path,
        "seconds": State.roi_frame_seconds,
        "video": get_video_info(State.video_path),
    })

@app.route('/api/set_roi', methods=['POST'])
def api_set_roi():
    data = request.json
    points = data.get('points', [])
    if len(points) < 3:
        return jsonify({"error": "Need at least 3 points"}), 400
    
    State.roi_poly = np.array(points, np.int32).reshape((-1, 1, 2))
    State.is_processing = True
    State.is_finished = False
    State.progress = 0
    State.current_fuel_count = 0
    State.robot_crops = {}
    State.robot_scores = defaultdict(int)
    State.total_scored_fuel = 0
    
    # Start background processing thread
    threading.Thread(target=process_video_task).start()
    return jsonify({"success": True})

def process_video_task():
    cap = cv2.VideoCapture(State.video_path if os.path.exists(State.video_path) else 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    State.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if State.total_frames <= 0: State.total_frames = 1
    
    # Write directly to mp4 using avc1 codec (H264)
    out_path = 'static/output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    fuel_history = defaultdict(list)
    robot_history = defaultdict(list)
    airborne_fuel_ids = set()
    scored_fuel_ids = set()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        State.progress = frame_count
        display_frame = frame.copy()

        if State.roi_poly is not None:
            cv2.polylines(display_frame, [State.roi_poly], True, (255, 255, 255), 2)

        robot_boxes_list, robot_ids_list = [], []
        fuel_boxes_list, fuel_ids_list = [], []

        if unified_model:
            results = unified_model.track(
                frame, persist=True, verbose=False, imgsz=320
            )
            
            robot_cls = 0 if unified_model.model_name in ['yolo11n.yaml', 'yolov8n.yaml'] else next((k for k, v in unified_model.names.items() if v == 'robot'), 1)
            fuel_cls = 32 if unified_model.model_name in ['yolo11n.yaml', 'yolov8n.yaml'] else next((k for k, v in unified_model.names.items() if v == 'fuel'), 0)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for b, t_id, c in zip(boxes, ids, classes):
                    if c == robot_cls:
                        robot_boxes_list.append(b)
                        robot_ids_list.append(t_id)
                    elif c == fuel_cls:
                        fuel_boxes_list.append(b)
                        fuel_ids_list.append(t_id)
        else:
            fuel_results = fuel_model.track(
                frame, persist=True, verbose=False, imgsz=320, classes=[32] if fuel_model.model_name in ['yolo11n.yaml', 'yolov8n.yaml'] else None
            )
            robot_results = robot_model.track(
                frame, persist=True, verbose=False, imgsz=320, classes=[0] if robot_model.model_name in ['yolo11n.yaml', 'yolov8n.yaml'] else None
            )
            
            if robot_results[0].boxes.id is not None:
                boxes = robot_results[0].boxes.xyxy.cpu().numpy()
                ids = robot_results[0].boxes.id.cpu().numpy().astype(int)
                for b, t_id in zip(boxes, ids):
                    robot_boxes_list.append(b)
                    robot_ids_list.append(t_id)
                    
            if fuel_results[0].boxes.id is not None:
                boxes = fuel_results[0].boxes.xyxy.cpu().numpy()
                ids = fuel_results[0].boxes.id.cpu().numpy().astype(int)
                for b, t_id in zip(boxes, ids):
                    fuel_boxes_list.append(b)
                    fuel_ids_list.append(t_id)

        State.current_fuel_count = len(fuel_boxes_list)

        # Process Robots
        for box, r_id in zip(robot_boxes_list, robot_ids_list):
            rx1, ry1, rx2, ry2 = map(int, box)
            robot_history[r_id].append({'frame': frame_count, 'bbox': (rx1, ry1, rx2, ry2)})
            
            if r_id not in State.robot_crops or frame_count % 30 == 0:
                crop = frame[max(0, ry1-10):ry2+10, max(0, rx1-10):rx2+10]
                if crop.size > 0:
                    State.robot_crops[r_id] = crop

            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (128, 0, 128), 3)
            cv2.putText(display_frame, f"Robot {r_id}", (rx1, ry1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)

        # Process FUEL
        for box, f_id in zip(fuel_boxes_list, fuel_ids_list):
            fx1, fy1, fx2, fy2 = map(int, box)
            cx = int((fx1 + fx2) / 2)
            cy = int((fy1 + fy2) / 2)
            fuel_history[f_id].append({'frame': frame_count, 'x': cx, 'y': cy, 'bbox': (fx1, fy1, fx2, fy2)})
            
            if f_id not in airborne_fuel_ids:
                if is_airborne(fuel_history[f_id]):
                    airborne_fuel_ids.add(f_id)

            color = (0, 0, 255)
            cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), color, 2)
            
            if f_id in airborne_fuel_ids:
                color = (255, 0, 0)
                status_text = f"FUEL {f_id}"
                
                if f_id not in scored_fuel_ids and State.roi_poly is not None:
                    inside = cv2.pointPolygonTest(State.roi_poly, (cx, cy), False) >= 0
                    if inside:
                        scored_fuel_ids.add(f_id)
                        State.total_scored_fuel += 1
                        source_robot_id = calculate_trajectory_and_attribute(f_id, fuel_history[f_id], robot_history)
                        if source_robot_id is not None:
                            State.robot_scores[source_robot_id] += 1

                if f_id in scored_fuel_ids:
                    color = (0, 255, 0)
                    status_text = "SCORED"
                    
                cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), color, 2)
                cv2.putText(display_frame, status_text, (fx1, fy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                pts = np.array([[pt['x'], pt['y']] for pt in fuel_history[f_id]], np.int32)
                cv2.polylines(display_frame, [pts], False, color, 1)

        cv2.putText(display_frame, f"Scored FUEL: {State.total_scored_fuel}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        out.write(display_frame)

    cap.release()
    out.release()
    
    State.is_processing = False
    State.is_finished = True

@app.route('/api/status')
def api_status():
    return jsonify({
        "is_processing": State.is_processing,
        "is_finished": State.is_finished,
        "progress": State.progress,
        "total_frames": State.total_frames,
        "total_scored": State.total_scored_fuel,
        "current_fuel_count": State.current_fuel_count
    })

@app.route('/api/results')
def api_results():
    crops_b64 = {}
    for r_id, crop in State.robot_crops.items():
        _, buffer = cv2.imencode('.jpg', crop)
        crops_b64[r_id] = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "scores": State.robot_scores,
        "crops": crops_b64,
        "total_scored": State.total_scored_fuel
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
