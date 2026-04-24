import os
import cv2
import base64
import threading
import numpy as np
from flask import Flask, render_template, request, Response, jsonify
from collections import defaultdict
from ultralytics import YOLO

app = Flask(__name__)
os.makedirs('static/crops', exist_ok=True)

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

# Initialize Unified Model
try:
    # Replace with the path to your new unified model
    unified_model = YOLO('unified.pt') 
except Exception:
    print("ERROR: Could not load unified model, defaulting to YOLOv11.")
    unified_model = YOLO('yolov11n.pt')

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
    cap = cv2.VideoCapture(State.video_path if os.path.exists(State.video_path) else 0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"error": "Could not read video"}), 500
        
    State.first_frame = frame
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"image": f"data:image/jpeg;base64,{jpg_as_text}"})

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

        # Single inference call for unified model
        results = unified_model.track(
            frame, persist=True, verbose=False, imgsz=320, device="0"
        )
        
        # Determine class indices dynamically
        # If using yolov8n fallback: 0 is person (robot), 32 is sports ball (fuel)
        robot_cls = 0 if unified_model.model_name == 'yolov8n.yaml' else next((k for k, v in unified_model.names.items() if v == 'robot'), 0)
        fuel_cls = 32 if unified_model.model_name == 'yolov8n.yaml' else next((k for k, v in unified_model.names.items() if v == 'fuel'), 1)

        robot_boxes_list, robot_ids_list = [], []
        fuel_boxes_list, fuel_ids_list = [], []

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
    app.run(host='0.0.0.0', port=5000, debug=True)
