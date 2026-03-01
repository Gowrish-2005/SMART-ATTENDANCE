# run_session_fixed.py
"""
Single-run Smart Attendance:
 - Uses a single VideoCapture for face recognition + MediaPipe gaze/attention.
 - Writes attendance.csv (timestamp, name, attention_score, status).
 - Sends SMS to absentees if Twilio env vars are set and 'twilio' package installed.
Config at top of file.
"""

import os
import sys
import time
import csv
from datetime import datetime
import traceback

import cv2
import numpy as np
import face_recognition

# Optional imports - handled gracefully
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False
    mp = None

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TwilioClient = None
    TWILIO_AVAILABLE = False

# ---------- Config ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINED_PATH = os.path.join(ROOT, "trained_faces.pkl")  # must exist (run train_model.py)
STUDENTS_CSV = os.path.join(ROOT, "students.csv")
ATTENDANCE_CSV = os.path.join(ROOT, "attendance.csv")

DIST_THRESHOLD = 0.45    # lower -> stricter match
CONFIRM_FRAMES = 3       # consecutive frames required to confirm present
SESSION_DURATION = 30    # seconds; set longer for classroom
CAMERA_INDEX = 0   

CAMERA_BACKENDS = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]  # Windows-friendly; v4l2 on Linux may be used
PROCESS_EVERY_N_FRAMES = 1  # process each frame for real-time
# -------------------------

def load_trained():
    import pickle
    if not os.path.exists(TRAINED_PATH):
        raise FileNotFoundError(f"Trained faces file not found: {TRAINED_PATH}")
    with open(TRAINED_PATH, "rb") as f:
        data = pickle.load(f)
    encs = np.array(data.get("encodings", []))
    names = np.array(data.get("names", []))
    return encs, names


def load_roster():
    roster = {}
    if not os.path.exists(STUDENTS_CSV):
        print(f"[WARN] students.csv not found at {STUDENTS_CSV}. Continuing without roster (SMS disabled).")
        return roster
    with open(STUDENTS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name") or row.get("name") or row.get("NAME")
            phone = row.get("Phone") or row.get("Phone Number") or row.get("PhoneNumber") or row.get("phone")
            if name:
                roster[name.strip()] = (phone.strip() if phone else "")
    return roster

def init_mediapipe():
    if not MEDIAPIPE_AVAILABLE:
        print("[WARN] MediaPipe not installed; attention scoring will be disabled.")
        return None, None
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=5,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    return face_mesh, mp_drawing

def attention_score_from_landmarks(landmarks, image_w, image_h):
    try:
        left_iris = np.array([(landmark.x * image_w, landmark.y * image_h) for landmark in [landmarks[i] for i in (474,475,476,477)]])
        right_iris = np.array([(landmark.x * image_w, landmark.y * image_h) for landmark in [landmarks[i] for i in (469,470,471,472)]])
        eye_center = (left_iris.mean(axis=0) + right_iris.mean(axis=0)) / 2.0
        screen_center = np.array([image_w/2.0, image_h/2.0])
        dist = np.linalg.norm(eye_center - screen_center)
        diag = np.linalg.norm([image_w, image_h])
        score = max(0.0, 1.0 - (dist / (diag * 0.35)))
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.5

def try_open_camera():
    cap = None
    for backend in CAMERA_BACKENDS:
        try:
            cap = cv2.VideoCapture(CAMERA_INDEX, backend)
            if cap is None or not cap.isOpened():
                if cap:
                    cap.release()
                cap = None
                continue
            # test read
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                cap = None
                continue
            print(f"[INFO] Camera opened with backend {backend}.")
            return cap
        except Exception:
            if cap:
                try: cap.release()
                except: pass
            cap = None
    # fallback: try default without backend
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap and cap.isOpened():
        print("[INFO] Camera opened with default backend.")
        return cap
    raise RuntimeError("Unable to open camera. Ensure camera is connected and not used by another app.")

def send_sms_list(absentees, roster):
    if not TWILIO_AVAILABLE:
        print("[WARN] Twilio not installed. Install 'twilio' to enable SMS.")
        return
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_phone = os.environ.get("TWILIO_FROM")
    if not sid or not token or not from_phone:
        print("[WARN] Twilio env vars not set. Skipping SMS.")
        return
    client = TwilioClient(sid, token)
    for name in absentees:
        phone = roster.get(name, "")
        if not phone:
            print(f"[WARN] No phone for {name}; skipping SMS")
            continue
        body = f"Dear Parent, your child {name} is marked ABSENT today."
        try:
            client.messages.create(body=body, from_=from_phone, to=phone)
            print(f"[SMS] Sent to {name} -> {phone}")
        except Exception as e:
            print(f"[ERROR] SMS failed for {name}: {e}")

def ensure_attendance_csv():
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "name", "attention_score", "status"])

def main():
    try:
        known_encodings, known_names = load_trained()
    except Exception as e:
        print("[ERROR] Could not load trained encodings:", e)
        return

    roster = load_roster()
    expected_names = set(roster.keys()) if roster else set(known_names)

    face_mesh, mp_drawing = init_mediapipe()
    ensure_attendance_csv()

    # state
    seen_counts = {}
    attention_scores = {}
    confirmed_present = set()
    frame_count = 0
    start_time = time.time()

    # open camera once
    try:
        cap = try_open_camera()
    except Exception as e:
        print("[ERROR] Camera open failed:", e)
        return

    print("[INFO] Session started. Press 'q' to end early.")
    # set resolution
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except Exception:
        pass

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Failed to read frame.")
            break
        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Face detection & encoding
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            face_locations = face_recognition.face_locations(rgb, model="hog")
            face_encodings = face_recognition.face_encodings(rgb, face_locations)
        else:
            face_locations = []
            face_encodings = []

        # Mediapipe
        mp_results = None
        if face_mesh:
            mp_results = face_mesh.process(rgb)

        names_this_frame = []
        # match encodings -> names
        for enc in face_encodings:
            if len(known_encodings) == 0:
                names_this_frame.append("Unknown")
                continue
            dists = face_recognition.face_distance(known_encodings, enc)
            idx = int(np.argmin(dists))
            dist = float(dists[idx])
            if dist < DIST_THRESHOLD:
                pred_name = known_names[idx]
                # normalize to roster name (case-insensitive)
                match_name = None
                for r in expected_names:
                    if pred_name.strip().upper() == r.strip().upper():
                        match_name = r
                        break
                if match_name:
                    names_this_frame.append(match_name)
                else:
                    names_this_frame.append(pred_name)
            else:
                names_this_frame.append("Unknown")

        # compute attention for faces (if mediapipe landmarks available)
        if mp_results and mp_results.multi_face_landmarks:
            # simple mapping: iterate faces and get attention from the corresponding mp face index
            for i, landmarks in enumerate(mp_results.multi_face_landmarks):
                # we won't map by bounding box; instead compute attention per face index
                attn = attention_score_from_landmarks(landmarks.landmark, w, h)
                # store in attention list by index
                # We can't reliably map encodings <-> mediapipe faces without additional matching;
                # for simplicity we will attach attention to the first matched name if counts align.
                # (This is a pragmatic approach used in many lightweight systems.)
                attention_scores[f"mp_{i}"] = attn

        # Update seen/confirmed by matching face_locations -> names_this_frame
        # If numbers mismatch, we try to match sequentially (best-effort)
        for idx, (loc, name) in enumerate(zip(face_locations, names_this_frame)):
            top, right, bottom, left = loc
            if name != "Unknown":
                # try to obtain an attention numeric value:
                attn_val = None
                # if mediapipe had results and same counts, use that
                if mp_results and mp_results.multi_face_landmarks and idx < len(mp_results.multi_face_landmarks):
                    attn_val = attention_score_from_landmarks(mp_results.multi_face_landmarks[idx].landmark, w, h)
                else:
                    # fallback: default 0.5
                    attn_val = attention_scores.get(name, 0.5)
                attention_scores[name] = attn_val

                seen_counts[name] = seen_counts.get(name, 0) + 1
                # confirm presence
                if seen_counts[name] >= CONFIRM_FRAMES and name not in confirmed_present:
                    confirmed_present.add(name)
                    # append to attendance csv
                    with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            datetime.now().isoformat(timespec="seconds"),
                            name,
                            f"{attn_val:.2f}",
                            "PRESENT"
                        ])
                    print(f"[ATTENDANCE] {name} marked PRESENT (att={attn_val:.2f})")
            else:
                # unknown face: ignore for attendance, but draw red rectangle
                pass

            # draw bounding box & label
            label = name
            if name != "Unknown":
                label = f"{name} ({attention_scores.get(name, 0.5):.2f})"
            color = (0,255,0) if name != "Unknown" else (0,0,255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # optionally draw mediapipe landmarks
        if mp_results and mp_results.multi_face_landmarks and mp_drawing:
            for fm in mp_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, fm, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

        # overlays
        elapsed = time.time() - start_time
        cv2.putText(frame, f"Time: {int(elapsed)}s / {SESSION_DURATION}s", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Present: {len(confirmed_present)}/{len(expected_names)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Smart Attendance (single-run)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] User ended session.")
            break
        if elapsed >= SESSION_DURATION:
            print("[INFO] Session duration reached.")
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

    # finalize: compute absentees from expected_names (roster-based) or trained names
    if expected_names:
        present = set(confirmed_present)
        absentees = sorted(list(expected_names - present))
    else:
        # fallback: if roster not provided, derive from known_names
        names_set = set([n for n in np.unique(known_names)])
        absentees = sorted(list(names_set - set(confirmed_present)))

    print("\n[SUMMARY]")
    print("Present:", sorted(list(confirmed_present)))
    print("Absent:", absentees)

    # Send SMS if configured
    if absentees:
        if TWILIO_AVAILABLE and os.environ.get("TWILIO_ACCOUNT_SID") and os.environ.get("TWILIO_AUTH_TOKEN") and os.environ.get("TWILIO_FROM"):
            print("[INFO] Sending SMS to absentees...")
            send_sms_list(absentees, roster)
        else:
            print("[INFO] Twilio not configured or not installed; SMS skipped.")

    print("[COMPLETE] Session finished.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL] Exception in run_session_fixed.py:")
        traceback.print_exc()
