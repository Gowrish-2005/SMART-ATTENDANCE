import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

import cv2
import face_recognition
import numpy as np
import pickle
import csv
from datetime import datetime

def recognize_students():
    # Load trained encodings
    with open("trained_faces.pkl", "rb") as f:
        data = pickle.load(f)

    known_encodings = np.array(data["encodings"])
    known_names = np.array(data["names"])

    # Load master student list (get unique names)
    students = list(set(known_names))
    present_students = []

    # Initialize webcam
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not video_capture.isOpened():
        print("❌ Error: Could not open camera. Please check if camera is connected.")
        return [], []
    
    # Set camera resolution for better performance
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    attendance_file = "attendance.csv"

    # Ensure attendance file exists
    try:
        open(attendance_file, "x", newline="").close()
    except FileExistsError:
        pass

    def mark_attendance(name):
        with open(attendance_file, "r+", newline="") as f:
            data = list(csv.reader(f))
            name_list = [row[0] for row in data]

            if name not in name_list:
                now = datetime.now()
                dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
                writer = csv.writer(f)
                writer.writerow([name, dt_string])
                print(f"✅ Marked attendance for {name}")

    print(f"📷 Starting camera... Press 'q' to quit.")
    print(f"📊 Loaded {len(known_encodings)} face encodings for {len(set(known_names))} students")
    print(f"👥 Students: {', '.join(set(known_names))}")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("⚠️  Warning: Could not read frame from camera")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            
            if len(known_encodings) > 0:
                # Use face_distance for better accuracy control
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                best_match_distance = face_distances[best_match_index]
                
                # Use tolerance of 0.45 for better accuracy (lower = stricter)
                if best_match_distance < 0.45:
                    name = known_names[best_match_index]
                    if name not in present_students:
                        present_students.append(name)
                        mark_attendance(name)

            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Attendance Recognition", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    # Compute absentees
    absent_students = [s for s in students if s not in present_students]

    print("🟢 Attendance recording finished.")
    print("Present students:", present_students)
    print("Absent students:", absent_students)

    return present_students, absent_students

if __name__ == "__main__":
    recognize_students()
