import cv2
import dlib
import numpy as np
import os

# Path to the landmark model (keep as absolute or use env var)
PREDICTOR_PATH = os.getenv(
    "SHAPE_PREDICTOR_PATH",
    r"C:\Users\ADMIN\OneDrive\Desktop\SMART ATTENDANCE\shape_predictor_68_face_landmarks.dat",
)


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_eye_region(landmarks, left=True):
    if left:
        points = [36, 37, 38, 39, 40, 41]
    else:
        points = [42, 43, 44, 45, 46, 47]
    region = np.array([(landmarks.part(p).x, landmarks.part(p).y) for p in points], np.int32)
    return region


def get_gaze_ratio(eye_points, facial_landmarks, frame_gray, left=True):
    region = get_eye_region(facial_landmarks, left)
    mask = np.zeros(frame_gray.shape, dtype=np.uint8)
    cv2.polylines(mask, [region], True, 255, 2)
    cv2.fillPoly(mask, [region], 255)
    eye = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)

    min_x = np.min(region[:, 0])
    max_x = np.max(region[:, 0])
    min_y = np.min(region[:, 1])
    max_y = np.max(region[:, 1])

    gray_eye = eye[min_y:max_y, min_x:max_x]
    if gray_eye.size == 0:
        return 1
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    h, w = threshold_eye.shape
    if w == 0:
        return 1
    left_side = threshold_eye[0:h, 0:int(w / 2)]
    right_side = threshold_eye[0:h, int(w / 2):w]

    left_white = cv2.countNonZero(left_side)
    right_white = cv2.countNonZero(right_side)

    if right_white == 0:
        gaze_ratio = 1
    else:
        gaze_ratio = left_white / right_white
    return gaze_ratio


def track_gaze():
    """Start gaze tracking from the webcam. Blocks until 'q' is pressed.

    This function was extracted so other modules (e.g., main_model.py) can import
    and call it without triggering the webcam immediately on import.
    """
    # Load face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
    except Exception as e:
        print(f"Error loading shape predictor from {PREDICTOR_PATH}: {e}")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Unable to open webcam for gaze tracking.")
        return

    print("👁️  Gaze tracking started — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            gaze_left = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray, True)
            gaze_right = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray, False)
            gaze_ratio = (gaze_left + gaze_right) / 2

            if gaze_ratio <= 0.8:
                gaze_text = "RIGHT"
            elif 0.8 < gaze_ratio < 1.5:
                gaze_text = "CENTER"
            else:
                gaze_text = "LEFT"

            cv2.putText(frame, f"Gaze: {gaze_text}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Gaze Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🔵 Gaze tracking ended.")


if __name__ == "__main__":
    track_gaze()



