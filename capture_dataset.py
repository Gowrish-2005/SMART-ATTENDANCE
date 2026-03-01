import argparse
import cv2
import os

# Accept a name argument so the script can be called from the web UI
parser = argparse.ArgumentParser(description="Capture face images for one person")
parser.add_argument("--name", help="Name of the person to capture")
parser.add_argument("--count", type=int, default=50, help="Number of images to capture automatically")
args = parser.parse_args()

# Step 1: Get person name
if args.name:
    name = args.name.strip()
else:
    name = input("Enter the person's name: ").strip()

# Step 2: Create folder for that person
dataset_dir = "dataset"
person_dir = os.path.join(dataset_dir, name)
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

TARGET_COUNT = args.count  # auto-stop after this many images

# Step 3: Start webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cam.isOpened():
    cam = cv2.VideoCapture(0)  # fallback
if not cam.isOpened():
    print("[ERROR] Could not open camera.")
    exit(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Capturing Faces - press ESC to stop")
print(f"[INFO] Capturing {TARGET_COUNT} images for '{name}'. Press ESC to stop early.")

count = 0
frame_num = 0

while count < TARGET_COUNT:
    ret, frame = cam.read()
    if not ret:
        print("[WARNING] Failed to grab frame")
        break

    frame_num += 1

    # Auto-save every 3rd frame (to get varied images without blur)
    if frame_num % 3 == 0:
        img_name = f"{name}_{count}.jpg"
        cv2.imwrite(os.path.join(person_dir, img_name), frame)
        count += 1
        print(f"[{count}/{TARGET_COUNT}] Saved {img_name}")

    # Overlay progress
    progress = f"Capturing {name}: {count}/{TARGET_COUNT}"
    cv2.putText(frame, progress, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Capturing Faces - press ESC to stop", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC to exit early
        print("[INFO] Stopped early by user.")
        break

# Step 4: Release camera and close window
cam.release()
cv2.destroyAllWindows()
print(f"[DONE] Saved {count} images for '{name}' in '{person_dir}'")