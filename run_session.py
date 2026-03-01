import os
import sys
import csv
import time
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import pickle

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    Client = None

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- Configs for accuracy ----
DIST_THRESHOLD = 0.45  # tighter threshold for better accuracy
CONFIRM_FRAMES = 3     # frames needed to confirm identity (reduced for faster recognition)
PROCESS_EVERY_N = 1    # process every frame for real-time recognition
SESSION_DURATION = 30  # seconds to run session before finalizing attendance
USE_CAMERA = False    # Camera disabled

# ---- Load roster (expected people + phone numbers) ----
ROSTER_CSV = os.path.join(ROOT, "students.csv")
roster = {}
if os.path.exists(ROSTER_CSV):
    with open(ROSTER_CSV, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("Name", "").strip()
            phone = row.get("Phone", "").strip()
            if name:
                roster[name] = phone
else:
    print("[WARNING] students.csv not found. SMS will be skipped.")

expected_names = set(roster.keys())
present_names = set()  # Will be updated as faces are recognized

# ---- Load trained encodings ----
TRAINED_PATH = os.path.join(ROOT, "trained_faces.pkl")
if not os.path.exists(TRAINED_PATH):
    raise FileNotFoundError("trained_faces.pkl not found. Run 'Train model' first (option 2 in main.py).")

with open(TRAINED_PATH, "rb") as f:
    data = pickle.load(f)
known_encodings = np.array(data.get("encodings", []))
known_names = np.array(data.get("names", []))

# Normalize names to match roster (case-insensitive)
name_mapping = {}
for known_name in known_names:
    for roster_name in expected_names:
        if known_name.upper() == roster_name.upper():
            name_mapping[known_name] = roster_name
            break

# ---- Mediapipe for attention/gaze ----
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,  # Allow multiple faces
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def attention_score_from_landmarks(landmarks, image_w, image_h):
    """Calculate attention score based on gaze direction"""
    left_iris_idxs = [474, 475, 476, 477]
    right_iris_idxs = [469, 470, 471, 472]
    
    def center(idxs):
        pts = np.array([(landmarks[i].x * image_w, landmarks[i].y * image_h) for i in idxs], dtype=np.float32)
        return pts.mean(axis=0)
    
    try:
        eye_center = (center(left_iris_idxs) + center(right_iris_idxs)) / 2.0
        screen_center = np.array([image_w/2.0, image_h/2.0], dtype=np.float32)
        dist = np.linalg.norm(eye_center - screen_center)
        diag = np.linalg.norm([image_w, image_h])
        score = max(0.0, 1.0 - (dist/(diag*0.35)))
        return float(np.clip(score, 0.0, 1.0))
    except:
        return 0.5  # Default if calculation fails

# ---- Twilio setup ----
twilio_client = None
if TWILIO_AVAILABLE:
    ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
    AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM = os.environ.get("TWILIO_FROM", "")
    if ACCOUNT_SID and AUTH_TOKEN and TWILIO_FROM:
        try:
            twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)
            print("[SUCCESS] Twilio client initialized")
        except Exception as e:
            print(f"[WARNING] Twilio client init failed: {e}")
    else:
        print("[WARNING] Twilio credentials not set. SMS will be disabled.")
else:
    print("[WARNING] Twilio not installed. SMS functionality will be disabled.")

# ---- Attendance output ----
ATTENDANCE_CSV = os.path.join(ROOT, "attendance.csv")
if not os.path.exists(ATTENDANCE_CSV):
    with open(ATTENDANCE_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "name", "attention_score", "status"])

# ---- Tracking state ----
seen_counts = {}       # name -> consecutive frames seen
last_seen = {}         # name -> last frame number seen
attention_scores = {}  # name -> latest attention score
confirmed_present = set()  # Names confirmed as present

# ---- Initialize camera ----
cap = None

if USE_CAMERA:
    print("\n[INFO] Initializing camera...")
    
    # Try different backends and indices
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "Any")
    ]
    
    for backend_id, backend_name in backends:
        for idx in range(3):  # Try camera indices 0-2
            try:
                if backend_id == cv2.CAP_ANY:
                    cap = cv2.VideoCapture(idx)
                else:
                    cap = cv2.VideoCapture(idx, backend_id)
                
                if cap.isOpened():
                    # Test if we can read a frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"[SUCCESS] Camera opened at index {idx} using {backend_name}")
                        break
                    else:
                        cap.release()
                        cap = None
                else:
                    if cap:
                        cap.release()
                    cap = None
            except Exception as e:
                if cap:
                    cap.release()
                cap = None
                continue
        
        if cap and cap.isOpened():
            break
    
    if not cap or not cap.isOpened():
        print("[ERROR] Cannot open camera. Please check:")
        print("  1. Camera is connected and not being used by another application")
        print("  2. Camera drivers are installed")
        print("  3. Camera permissions are granted")
        print("\n[INFO] Exiting...")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"\n[INFO] Session started!")
    print(f"[INFO] Expected students: {sorted(expected_names)}")
    print(f"[INFO] Press 'q' to end session and send SMS to absentees")
    print(f"[INFO] Session will also auto-end after {SESSION_DURATION} seconds\n")
else:
    print("\n[INFO] Camera disabled (USE_CAMERA = False)")
    print(f"[INFO] Expected students: {sorted(expected_names)}")
    print(f"[INFO] Session will run for {SESSION_DURATION} seconds without camera\n")

# ---- Main video loop ----
frame_count = 0
start_time = time.time()
session_active = True

if USE_CAMERA:
    while session_active:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame")
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Face recognition every frame for real-time detection
        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, locations)
        
        names_this_frame = []
        for enc in encodings:
            if len(known_encodings) == 0:
                names_this_frame.append("Unknown")
                continue
            dists = face_recognition.face_distance(known_encodings, enc)
            idx = int(np.argmin(dists))
            dist = float(dists[idx])
            if dist < DIST_THRESHOLD:
                pred_name = known_names[idx]
                # Map to roster name if needed
                roster_name = name_mapping.get(pred_name, pred_name)
                if roster_name.upper() in [n.upper() for n in expected_names]:
                    names_this_frame.append(roster_name)
                else:
                    names_this_frame.append("Unknown")
            else:
                names_this_frame.append("Unknown")

        # Process MediaPipe for gaze tracking
        results = face_mesh.process(rgb)
        
        # Update tracking for recognized faces
        current_frame_names = set()
        for (top, right, bottom, left), name in zip(locations, names_this_frame):
            if name != "Unknown":
                current_frame_names.add(name)
                last_seen[name] = frame_count
                
                # Calculate attention for this face
                if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                    # Match face to MediaPipe landmark (simple: use first face)
                    face_idx = min(len(results.multi_face_landmarks) - 1, names_this_frame.index(name) if name in names_this_frame else 0)
                    if face_idx < len(results.multi_face_landmarks):
                        lms = results.multi_face_landmarks[face_idx].landmark
                        attn = attention_score_from_landmarks(lms, w, h)
                        attention_scores[name] = attn

        # Update seen counts
        for name in expected_names:
            if name in current_frame_names:
                seen_counts[name] = seen_counts.get(name, 0) + 1
                if seen_counts[name] >= CONFIRM_FRAMES and name not in confirmed_present:
                    confirmed_present.add(name)
                    present_names.add(name)
                    # Write to attendance CSV
                    with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
                        attn = attention_scores.get(name, 0.5)
                        csv.writer(f).writerow([
                            datetime.now().isoformat(timespec="seconds"),
                            name,
                            f"{attn:.2f}",
                            "PRESENT"
                        ])
                    print(f"[ATTENDANCE] {name} marked PRESENT (Attention: {attn:.2f})")
            else:
                # Reset count if not seen this frame
                if name in seen_counts:
                    seen_counts[name] = max(0, seen_counts[name] - 1)

        # Draw on frame
        for (top, right, bottom, left), name in zip(locations, names_this_frame):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Display name and status
            label = name
            if name != "Unknown":
                attn = attention_scores.get(name, 0.5)
                label = f"{name} (Att: {attn:.2f})"
                if name in confirmed_present:
                    label += " [PRESENT]"
            
            cv2.putText(frame, label, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw MediaPipe landmarks for attention visualization
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    None, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        # Display status overlay
        elapsed = time.time() - start_time
        remaining = max(0, SESSION_DURATION - elapsed)
        
        # Status text
        status_y = 30
        cv2.putText(frame, f"Session Time: {int(elapsed)}s / {SESSION_DURATION}s", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        status_y += 30
        cv2.putText(frame, f"Present: {len(confirmed_present)}/{len(expected_names)}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # List present students
        if confirmed_present:
            status_y += 25
            present_list = ", ".join(sorted(confirmed_present))
            if len(present_list) > 50:
                present_list = present_list[:50] + "..."
            cv2.putText(frame, f"Present: {present_list}", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Smart Attendance - Face Recognition & Gaze Tracking", frame)
        
        # Check for quit or timeout
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Session ended by user")
            session_active = False
        elif elapsed >= SESSION_DURATION:
            print(f"\n[INFO] Session duration ({SESSION_DURATION}s) reached")
            session_active = False

    # Cleanup
    if cap:
        cap.release()
    cv2.destroyAllWindows()
else:
    # Wait for session duration when camera is disabled
    print("[INFO] Waiting for session duration...")
    while session_active:
        elapsed = time.time() - start_time
        if elapsed >= SESSION_DURATION:
            print(f"\n[INFO] Session duration ({SESSION_DURATION}s) reached")
            session_active = False
        time.sleep(0.1)  # Small sleep to avoid busy waiting

# ---- Finalize attendance and send SMS ----
print("\n" + "="*60)
print("[ATTENDANCE SUMMARY]")
print("="*60)
print(f"Expected students: {len(expected_names)}")
print(f"Present: {len(confirmed_present)}")
print(f"Absent: {len(expected_names) - len(confirmed_present)}")
print()

if confirmed_present:
    print("[PRESENT STUDENTS]:")
    for name in sorted(confirmed_present):
        print(f"  - {name}")
    print()

absentees = sorted(list(expected_names - confirmed_present))
if absentees:
    print("[ABSENT STUDENTS]:")
    for name in absentees:
        print(f"  - {name}")
    print()
    
    # Send SMS to absentees only
    if twilio_client is None:
        print("[WARNING] Twilio not configured. SMS will not be sent.")
    else:
        print("[SENDING SMS TO ABSENTEES]...")
        print()
        for name in absentees:
            phone = roster.get(name)
            if not phone:
                print(f"[WARNING] No phone number for {name}, skipping SMS")
                continue
            
            body = f"Dear Parent, your child {name} is marked ABSENT today."
            try:
                from_number = os.environ.get("TWILIO_FROM", TWILIO_FROM)
                message = twilio_client.messages.create(
                    body=body,
                    from_=from_number,
                    to=phone
                )
                print(f"[SUCCESS] SMS sent to {name} -> {phone}")
            except Exception as e:
                print(f"[ERROR] Failed to send SMS to {name} ({phone}): {e}")
        print()
else:
    print("[INFO] All students are present! No SMS sent.")
    print()

print("[COMPLETE] Session ended successfully!")
print("="*60)
