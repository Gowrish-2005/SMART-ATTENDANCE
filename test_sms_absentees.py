import os
import sys
import csv
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("ERROR: Twilio not installed. Install with: pip install twilio")
    exit(1)

ROOT = os.path.dirname(os.path.abspath(__file__))

# Load roster
ROSTER_CSV = os.path.join(ROOT, "students.csv")
roster = {}
if os.path.exists(ROSTER_CSV):
    with open(ROSTER_CSV, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("Name", "").strip()
            phone = row.get("Phone", "").strip()
            if name and phone:
                roster[name] = phone
else:
    print("[ERROR] students.csv not found!")
    exit(1)

expected_names = set(roster.keys())
print(f"[INFO] Loaded {len(expected_names)} students from roster: {sorted(expected_names)}")

# Get Twilio credentials
ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.environ.get("TWILIO_FROM", "")

if not ACCOUNT_SID or not AUTH_TOKEN or not TWILIO_FROM:
    print("[WARN] Twilio credentials not set in environment variables.")
    print("[INFO] Setting from known credentials...")
    ACCOUNT_SID = "AC501f5106f6cb135b96ab5d0e752b3d9b"
    AUTH_TOKEN = "2a7ce47a677dc1c502c4fb2ed848f8f6"
    TWILIO_FROM = "+12176248707"

try:
    twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)
    print("[SUCCESS] Twilio client initialized")
except Exception as e:
    print(f"[ERROR] Failed to initialize Twilio client: {e}")
    exit(1)

# Simulate: No one was present (all absent)
present_names = set()  # Empty set = no one attended
absentees = sorted(list(expected_names - present_names))

print(f"\n[ATTENDANCE SUMMARY]")
print(f"   Expected: {len(expected_names)}")
print(f"   Present: {len(present_names)}")
print(f"   Absent: {len(absentees)}")

if absentees:
    print(f"\n[SENDING SMS] Sending SMS to {len(absentees)} absentees...\n")
    for name in absentees:
        phone = roster.get(name)
        if not phone:
            print(f"[WARN] No phone number for {name}, skipping...")
            continue
        
        body = f"Dear Parent, your child {name} is marked absent today."
        try:
            message = twilio_client.messages.create(
                body=body,
                from_=TWILIO_FROM,
                to=phone
            )
            print(f"[SUCCESS] SMS sent to {name} -> {phone}")
        except Exception as e:
            print(f"[ERROR] Failed to send SMS to {name} ({phone}): {e}")
    
    print(f"\n[COMPLETE] SMS sending process completed!")
else:
    print("\n[INFO] No absentees. All students are present!")

