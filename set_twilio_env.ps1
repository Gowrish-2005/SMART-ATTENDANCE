# PowerShell script to set Twilio environment variables
# This script will prompt you to enter your Twilio credentials

Write-Host "=== Twilio Credentials Setup ===" -ForegroundColor Cyan
Write-Host ""

# Prompt for Account SID
$accountSid = Read-Host "Enter your Twilio Account SID"
if ([string]::IsNullOrWhiteSpace($accountSid)) {
    Write-Host "Using saved Account SID..." -ForegroundColor Yellow
    $accountSid = "AC501f5106f6cb135b96ab5d0e752b3d9b"
}

# Prompt for Auth Token
$authToken = Read-Host "Enter your Twilio Auth Token"
if ([string]::IsNullOrWhiteSpace($authToken)) {
    Write-Host "Using saved Auth Token..." -ForegroundColor Yellow
    $authToken = "4ebf7497d4c0493c4b7a89e05d822706"
}

# Prompt for Twilio Phone Number
$fromNumber = Read-Host "Enter your Twilio Phone Number (e.g., +12176248707)"
if ([string]::IsNullOrWhiteSpace($fromNumber)) {
    Write-Host "Using saved Phone Number..." -ForegroundColor Yellow
    $fromNumber = "+12176248707"
}

# Set environment variables
$env:TWILIO_ACCOUNT_SID = $accountSid
$env:TWILIO_AUTH_TOKEN = $authToken
$env:TWILIO_FROM = $fromNumber

Write-Host ""
Write-Host "Twilio environment variables set for this session:" -ForegroundColor Green
Write-Host "TWILIO_ACCOUNT_SID: $env:TWILIO_ACCOUNT_SID"
Write-Host "TWILIO_FROM: $env:TWILIO_FROM"
Write-Host "AUTH_TOKEN: (hidden)" -ForegroundColor Gray
Write-Host ""
Write-Host "You can now run: .\venv\Scripts\python.exe .\run_session.py" -ForegroundColor Cyan
