import face_recognition
import os
import cv2
import numpy as np
import pickle

dataset_dir = "dataset"
encodings = []
names = []

print("🔍 Scanning dataset folders...")

for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"📂 Processing {person_name}...")
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        # Load the image
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)

        # Proceed only if a face is found
        if len(face_locations) > 0:
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            encodings.append(face_encoding)
            names.append(person_name)
        else:
            print(f"⚠️ No face found in {image_name}, skipping...")

# Save encodings and names in a pickle file
data = {"encodings": encodings, "names": names}
with open("trained_faces.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Training complete!")
print(f"Total people trained: {len(set(names))}")


