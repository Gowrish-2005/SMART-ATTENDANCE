import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def venv_python():
    return os.path.join(ROOT, "venv", "Scripts", "python.exe")

def run(script):
    py = venv_python() if os.path.exists(os.path.join(ROOT, "venv", "Scripts", "python.exe")) else sys.executable
    subprocess.run([py, os.path.join(ROOT, script)], check=False)

def menu():
    print("\n=== SMART ATTENDANCE ===")
    print("1) Capture dataset (add new person images)")
    print("2) Train model")
    print("3) Recognize attendance")
    print("4) Run session (recognize + gaze + SMS)")
    print("5) Send SMS to absentees (from attendance.xlsx)")
    print("0) Exit")
    return input("Choose: ").strip()

if __name__ == "__main__":
    while True:
        choice = menu()
        if choice == "1":
            run("capture_dataset.py")
        elif choice == "2":
            run("train_model.py")
        elif choice == "3":
            run("recognize_attendance.py")
        elif choice == "4":
            run("run_session.py")
        elif choice == "5":
            run("sms_alert.py")
        elif choice == "0":
            break
        else:
            print("Invalid choice.")