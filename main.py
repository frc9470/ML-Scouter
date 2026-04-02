"""
FRC 2026 REBUILT Vision Tool
----------------------------
This application implements a 3-phase computer vision pipeline to analyze 
FIRST Robotics Competition match videos.

Phase 1: Airborne FUEL tracking and Polygonal ROI scoring validation.
Phase 2: Simultaneous Robot tracking.
Phase 3: Kinematic trajectory back-calculation for score attribution.

Dependencies:
    pip install opencv-python numpy ultralytics
"""

import cv2
import numpy as np
import sys
import os
from collections import defaultdict
from ultralytics import YOLO

# --- GLOBAL VARIABLES FOR GUI ---
roi_vertices = []
window_name = "ROI Selection - FRC REBUILT"

def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse events to construct the Region of Interest (ROI) polygon.
    Appends clicked (x, y) coordinates to the global roi_vertices list.
    """
    global roi_vertices
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_vertices.append((x, y))

def is_airborne(history, min_frames=3, velocity_threshold=2.0):
    """
    Kinematic heuristic to determine if a tracked object is airborne.
    Calculates pixel velocity (dy/dt). A negative value means it's moving UP.
    """
    if len(history) < min_frames:
        return False
    
    # Compare current Y to an older Y
    current_y = history[-1]['y']
    older_y = history[-min_frames]['y']
    
    # If it has moved UP significantly over the frame window, it's airborne
    if (older_y - current_y) > velocity_threshold:
        return True
    return False

def calculate_trajectory_and_attribute(fuel_id, fuel_history, robot_histories):
    """
    Phase III: Polynomial regression and spatial intersection.
    Fits a curve to the FUEL's trajectory and finds the closest robot at the launch frame.
    """
    if len(fuel_history) < 3:
        return None # Not enough data to map a trajectory
    
    # 1. Extract historical coordinates
    frames = [pt['frame'] for pt in fuel_history]
    X = np.array([pt['x'] for pt in fuel_history])
    Y = np.array([pt['y'] for pt in fuel_history])
    
    launch_frame = frames[0]
    launch_x = X[0]
    
    # 2. Fit 2nd-degree polynomial (Parabola): y = ax^2 + bx + c
    # We use a try-except to catch RankWarnings if the ball goes straight up (x doesn't change)
    try:
        with np.errstate(all='ignore'):
            coeffs = np.polyfit(X, Y, 2)
            # Evaluate origin Y at the launch X coordinate
            origin_y = np.polyval(coeffs, launch_x)
    except Exception:
        # Fallback to straight line or raw launch coordinate if fit fails
        origin_y = Y[0]
        
    origin_point = (launch_x, origin_y)
    
    # 3. Spatial Intersection (Distance Heuristic)
    closest_robot_id = None
    min_distance = float('inf')
    
    for robot_id, rob_history in robot_histories.items():
        # Find where this robot was during the launch frame
        rob_state_at_launch = next((state for state in rob_history if state['frame'] == launch_frame), None)
        
        if rob_state_at_launch:
            rx1, ry1, rx2, ry2 = rob_state_at_launch['bbox']
            # Top-center of the robot's bounding box
            top_center_x = (rx1 + rx2) / 2.0
            top_center_y = ry1
            
            # Euclidean distance from calculated origin to robot top-center
            dist = np.sqrt((origin_point[0] - top_center_x)**2 + (origin_point[1] - top_center_y)**2)
            
            if dist < min_distance:
                min_distance = dist
                closest_robot_id = robot_id
                
    return closest_robot_id

def main():
    video_path = "match_video.mp4" # Replace with your match video path
    
    # Attempt to open video
    cap = cv2.VideoCapture(video_path if os.path.exists(video_path) else 0)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}.")
        print("Please provide a valid match video file named 'match_video.mp4' or connect a webcam.")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        sys.exit(1)

    # --- PHASE I: Interactive ROI Selection ---
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- ROI SELECTION ---")
    print("Click on the video window to trace the hexagonal HUB funnel.")
    print("Press 'ENTER' to confirm the polygon and begin analysis.")
    
    while True:
        display_frame = first_frame.copy()
        
        # Draw the polygon as the user clicks
        if len(roi_vertices) > 0:
            for i in range(len(roi_vertices)):
                cv2.circle(display_frame, roi_vertices[i], 4, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(display_frame, roi_vertices[i-1], roi_vertices[i], (0, 255, 255), 2)
            if len(roi_vertices) > 2:
                # Close the shape visually
                cv2.line(display_frame, roi_vertices[-1], roi_vertices[0], (0, 255, 255), 2)
                
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13: # ASCII 13 is ENTER
            if len(roi_vertices) >= 3:
                print("ROI Confirmed. Initializing Neural Networks...")
                break
            else:
                print("Please select at least 3 points for a valid polygon.")
                
    cv2.destroyWindow(window_name)
    roi_poly = np.array(roi_vertices, np.int32).reshape((-1, 1, 2))

    # --- PHASE II: Model Initialization ---
    # Load custom Roboflow models if they exist, otherwise fallback to standard YOLOv8n
    try:
        fuel_model = YOLO('fuel_model_old.pt')
    except Exception:
        print("\n[WARNING] Custom 'fuel_model.pt' not found.")
        print("Falling back to standard YOLOv8n for prototype demonstration.")
        fuel_model = YOLO('yolov8n.pt')

    try:
        robot_model = YOLO('robot_model.pt')
    except Exception:
        print("\n[WARNING] Custom 'robot_model.pt' not found.")
        print("Falling back to standard YOLOv8n for prototype demonstration.")
        robot_model = YOLO('yolov8n.pt')

    # Data structures for tracking and scoring
    fuel_history = defaultdict(list)
    robot_history = defaultdict(list)
    airborne_fuel_ids = set()
    scored_fuel_ids = set()
    
    # Analytics
    robot_scores = defaultdict(int)
    robot_crops = {}
    total_scored_fuel = 0
    frame_count = 0

    print("\n--- COMMENCING MATCH ANALYSIS ---")
    print("Controls during playback:")
    print("  'p' - Pause/Resume")
    print("  'q' - Quit early")

    # Main Playback Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        display_frame = frame.copy()

        # Draw the ROI Polygon on the display frame
        cv2.polylines(display_frame, [roi_poly], True, (255, 255, 255), 2)

        # 1. Run inference and tracking on FUEL
        # (For fallback yolov8n, classes=[32] filters for sports balls)
        fuel_results = fuel_model.track(frame, persist=True, verbose=False, classes=[32] if fuel_model.model_name == 'yolov8n.yaml' else None)
        
        # 2. Run inference and tracking on Robots
        # (For fallback yolov8n, classes=[0] filters for people as a stand-in for robots)
        robot_results = robot_model.track(frame, persist=True, verbose=False, classes=[0] if robot_model.model_name == 'yolov8n.yaml' else None)

        # --- Process Robots ---
        if robot_results[0].boxes.id is not None:
            robot_boxes = robot_results[0].boxes.xyxy.cpu().numpy()
            robot_ids = robot_results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, r_id in zip(robot_boxes, robot_ids):
                rx1, ry1, rx2, ry2 = map(int, box)
                robot_history[r_id].append({
                    'frame': frame_count,
                    'bbox': (rx1, ry1, rx2, ry2)
                })
                
                # Save a clean crop of the robot for end-of-match Team ID mapping
                if r_id not in robot_crops or frame_count % 30 == 0:
                    # Keep a crop with a small margin
                    crop = frame[max(0, ry1-10):ry2+10, max(0, rx1-10):rx2+10]
                    if crop.size > 0:
                        robot_crops[r_id] = crop

                # Draw Purple Box for Robots
                cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (128, 0, 128), 3)
                cv2.putText(display_frame, f"Robot {r_id}", (rx1, ry1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)

        # --- Process FUEL ---
        if fuel_results[0].boxes.id is not None:
            fuel_boxes = fuel_results[0].boxes.xyxy.cpu().numpy()
            fuel_ids = fuel_results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, f_id in zip(fuel_boxes, fuel_ids):
                fx1, fy1, fx2, fy2 = map(int, box)
                cx = int((fx1 + fx2) / 2)
                cy = int((fy1 + fy2) / 2)
                
                fuel_history[f_id].append({
                    'frame': frame_count, 'x': cx, 'y': cy, 'bbox': (fx1, fy1, fx2, fy2)
                })
                
                # Check Airborne Status (Kinematic Filter)
                if f_id not in airborne_fuel_ids:
                    if is_airborne(fuel_history[f_id]):
                        airborne_fuel_ids.add(f_id)
                
                # Only process FUEL we consider airborne
                if f_id in airborne_fuel_ids:
                    color = (255, 0, 0) # Default Blue for tracking
                    status_text = f"FUEL {f_id}"
                    
                    # Point-in-Polygon validation for scoring
                    if f_id not in scored_fuel_ids:
                        inside = cv2.pointPolygonTest(roi_poly, (cx, cy), False) >= 0
                        
                        if inside:
                            scored_fuel_ids.add(f_id)
                            total_scored_fuel += 1
                            
                            # PHASE III: Trajectory and Attribution trigger
                            source_robot_id = calculate_trajectory_and_attribute(f_id, fuel_history[f_id], robot_history)
                            if source_robot_id is not None:
                                robot_scores[source_robot_id] += 1
                                print(f"[SCORING EVENT] FUEL {f_id} scored by Robot {source_robot_id}!")
                            else:
                                print(f"[SCORING EVENT] FUEL {f_id} scored (Unattributed).")

                    # Switch to Green if it has been marked as scored
                    if f_id in scored_fuel_ids:
                        color = (0, 255, 0) # Green
                        status_text = "SCORED"
                        
                    cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), color, 2)
                    cv2.putText(display_frame, status_text, (fx1, fy1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # Draw trajectory tail
                    pts = np.array([[pt['x'], pt['y']] for pt in fuel_history[f_id]], np.int32)
                    cv2.polylines(display_frame, [pts], False, color, 1)

        # Draw HUD
        cv2.putText(display_frame, f"Scored FUEL: {total_scored_fuel}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("FRC 2026 REBUILT Analysis Tool", display_frame)
        
        # User Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print("Playback PAUSED. Press 'p' again to resume.")
            while True:
                pause_key = cv2.waitKey(0) & 0xFF
                if pause_key == ord('p'):
                    print("Resuming...")
                    break
                if pause_key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

    # --- MATCH END AND POST-PROCESSING ---
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n================ MATCH ANALYSIS COMPLETE ================")
    print(f"Total Airborne FUEL Tracked: {len(airborne_fuel_ids)}")
    print(f"Total FUEL Successfully Scored in ROI: {total_scored_fuel}")
    print("---------------------------------------------------------")
    
    # Prompt user to attribute internal IDs to FRC Team Numbers
    final_team_scores = defaultdict(int)
    unattributed_score = total_scored_fuel - sum(robot_scores.values())
    
    if len(robot_scores) > 0:
        print("Commencing manual Team Number attribution for scoring robots...\n")
        
        for r_id, score in robot_scores.items():
            if r_id in robot_crops:
                # Show popup
                popup_name = f"Identify Robot {r_id}"
                cv2.imshow(popup_name, robot_crops[r_id])
                cv2.waitKey(100) # Ensure window renders before blocking input
                
                # Prompt in console
                try:
                    team_str = input(f"Please enter the FRC Team Number for the robot shown in the popup window (Internal Tracker ID: {r_id}): ")
                    team_num = int(team_str.strip())
                    final_team_scores[team_num] += score
                except ValueError:
                    print("Invalid input. Marking points as unattributed.")
                    unattributed_score += score
                    
                cv2.destroyWindow(popup_name)
            else:
                final_team_scores[f"Unknown ID {r_id}"] += score
                
    # Final Output Report
    print("\n--- Score Attribution by Team ---")
    for team, score in sorted(final_team_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"Team {team}: {score} FUEL Scored")
    print(f"Unattributed / Manual Interventions: {unattributed_score}")
    print("=========================================================\n")

if __name__ == "__main__":
    main()