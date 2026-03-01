from recognize_attendance import recognize_students
from gaze_tracking import track_gaze
from sms_alert import send_absent_sms

def main():
    print("Starting Smart Attendance System...")

    # Step 1: Recognize faces and mark attendance
    present_students, absent_students = recognize_students()

    # Step 2: Track gaze to verify attention (optional)
    
    track_gaze()


    # Step 3: Send SMS alerts to absent students' parents
    send_absent_sms(absent_students)

    print("Attendance process completed successfully!")

if __name__ == "__main__":
    main()
