import csv
import os

try:
    from twilio.rest import Client
    HAS_TWILIO = True
except ImportError:
    Client = None
    HAS_TWILIO = False
    # Print a clear one-time warning so the user knows why SMS won't send
    print("Warning: 'twilio' package not installed. SMS sending will be skipped."
          " Install with: python -m pip install twilio")


def send_absent_sms(absent_students):
    """Send SMS alerts for absent students.

    If the `twilio` package is missing the function will return early after
    printing a helpful message. Credentials can be provided via environment
    variables TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_PHONE to
    avoid hard-coding.
    """
    if not HAS_TWILIO:
        print("Twilio not available: skipping SMS send. To enable, run: python -m pip install twilio")
        return

    # Twilio credentials (prefer environment variables)
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    from_phone = os.getenv("TWILIO_FROM_PHONE", "+1XXXXXXXXXX")

    # Initialize Twilio client
    client = Client(account_sid, auth_token)

    # Load student contact details
    students_file = "students.csv"
    try:
        with open(students_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            students = list(reader)
    except FileNotFoundError:
        print(f"Students file not found: {students_file}. No SMS sent.")
        return

    for student in students:
        name = student.get("Name", "").strip()
        phone = student.get("Phone Number", "").strip()

        if name in absent_students:
            message_text = (
                f"Dear Parent, your child {name} was marked ABSENT today. "
                "Please ensure they attend regularly."
            )
            try:
                message = client.messages.create(
                    body=message_text,
                    from_=from_phone,
                    to=phone
                )
                print(f"📩 SMS sent successfully to {name}'s parent ({phone})")
            except Exception as e:
                print(f"⚠️ Failed to send SMS to {name} ({phone}): {e}")

    print("✅ SMS alert process completed.")
