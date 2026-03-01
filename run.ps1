# Single command script to run Smart Attendance System
$env:TWILIO_ACCOUNT_SID="AC501f5106f6cb135b96ab5d0e752b3d9b"
$env:TWILIO_AUTH_TOKEN="4ebf7497d4c0493c4b7a89e05d822706"
$env:TWILIO_FROM="+12176248707"
& .\venv\Scripts\python.exe .\run_session.py

