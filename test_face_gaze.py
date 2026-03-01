import cv2
import face_recognition
import mediapipe as mp

# Initialize mediapipe face mesh for gaze tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Open webcam
video_capture = cv2.VideoCapture(0)

print("✅ Webcam opened successfully. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Cannot access camera.")
        break

    # Convert frame to RGB for face recognition and mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face using face_recognition
    face_locations = face_recognition.face_locations(rgb_frame)

    # Draw rectangles for detected faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Gaze tracking using face mesh
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for id, lm in enumerate(face_landmarks.landmark[474:478]):  # eyes area
                ih, iw, _ = frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow('Face + Gaze Test', frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


