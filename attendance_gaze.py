# attendance_gaze.py
import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import os
import time
import csv
from datetime import datetime

# --- 1) Load known faces (encodings)
KNOWN_DIR = "data/known"
known_encodings = []
known_names = []

for name in os.listdir(KNOWN_DIR):
    person_dir = os.path.join(KNOWN_DIR, name)
    if not os.path.isdir(person_dir):
        continue
    for fname in os.listdir(person_dir):
        path = os.path.join(person_dir, fname)
        img = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(name)

# --- 2) Mediapipe setup for iris/face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  refine_landmarks=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- 3) Attendance store
attendance_file = "attendance_log.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","name","attention_score"])

# --- 4) Helpers
def recognize_face(frame_small):
    # frame_small: RGB small frame
    locations = face_recognition.face_locations(frame_small)
    encs = face_recognition.face_encodings(frame_small, locations)
    names = []
    for enc in encs:
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
        name = "Unknown"
        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]
        names.append(name)
    return locations, names, encs

def compute_attention(landmarks, image_w, image_h):
    # A simple heuristic: compute vector between iris center and image center.
    # Use MediaPipe refined iris landmarks indices (see docs). We'll use left/right iris centers as mean of iris landmarks.
    # landmarks: list of normalized landmarks
    left_iris_idxs = [474,475,476,477]   # mediapipe iris indexes (approx)
    right_iris_idxs = [469,470,471,472]
    def iris_center(idxs):
        pts = []
        for i in idxs:
            lm = landmarks[i]
            pts.append((lm.x * image_w, lm.y * image_h))
        pts = np.array(pts)
        return pts.mean(axis=0)
    left = iris_center(left_iris_idxs)
    right = iris_center(right_iris_idxs)
    eye_center = (left+right)/2.0
    screen_center = np.array([image_w/2.0, image_h/2.0])
    dist = np.linalg.norm(eye_center - screen_center)
    # normalize by diagonal
    diag = np.linalg.norm([image_w, image_h])
    score = max(0.0, 1.0 - (dist/(diag*0.35)))  # heuristic: 1.0 = looking center, 0 = far away
    return float(np.clip(score, 0.0, 1.0))

# --- 5) Run loop
cap = cv2.VideoCapture(0)
process_every_n_frames = 2
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(frame_rgb, (0,0), fx=0.5, fy=0.5)
    frame_count += 1

    # Face recognition every N frames (on smaller image)
    if frame_count % process_every_n_frames == 0:
        locations, names, encs = recognize_face(small)
    else:
        locations, names = [], []

    # MediaPipe face mesh (full res)
    results = face_mesh.process(frame_rgb)
    attention_score = None
    if results.multi_face_landmarks:
        # use first face
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        attention_score = compute_attention(landmarks, w, h)
        mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION)

    # Overlay recognized name(s) (scale locations back to full size)
    for (top, right, bottom, left), name in zip(locations, names):
        # locations are from small image -> scale by 2
        top *= 2; right *= 2; bottom *= 2; left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    if attention_score is not None:
        cv2.putText(frame, f"Attention: {attention_score:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.imshow("Attendance+Gaze", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and names:
        # quick save attendance snapshot of first detected student
        now = datetime.utcnow().isoformat()
        with open(attendance_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([now, names[0], attention_score if attention_score is not None else "NA"])

cap.release()
cv2.destroyAllWindows()