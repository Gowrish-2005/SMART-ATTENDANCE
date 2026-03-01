import os
import sys
import csv

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

try:
    from twilio.rest import Client
except ImportError:
    print("[ERROR] Twilio not installed. Install with: pip install twilio")
    exit(1)

ROOT = os.path.dirname(os.path.abspath(__file__))

# Get Twilio credentials from environment or use defaults
ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "AC501f5106f6cb135b96ab5d0e752b3d9b")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "4ebf7497d4c0493c4b7a89e05d822706")
TWILIO_FROM = os.environ.get("TWILIO_FROM", "+12176248707")

# Load students
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

print(f"[INFO] Loaded {len(roster)} students from roster")
print(f"[INFO] Sending SMS to all students...\n")

# Initialize Twilio
try:
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
except Exception as e:
    print(f"[ERROR] Failed to initialize Twilio client: {e}")
    exit(1)

# Send SMS to everyone
success_count = 0
fail_count = 0

for name, phone in roster.items():
    body = f"Dear Parent, your child {name} is marked absent today."
    try:
        message = client.messages.create(
            body=body,
            from_=TWILIO_FROM,
            to=phone
        )
        print(f"[SUCCESS] SMS sent to {name} -> {phone}")
        success_count += 1
    except Exception as e:
        print(f"[ERROR] Failed to send SMS to {name} ({phone}): {e}")
        fail_count += 1

print(f"\n[SUMMARY]")
print(f"   Success: {success_count}")
print(f"   Failed: {fail_count}")
print(f"   Total: {len(roster)}")

