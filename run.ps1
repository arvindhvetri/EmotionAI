Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting Facial Action Unit & Emotion Recognition App..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "URL: http://127.0.0.1:8001" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan

if (Test-Path ".\venv\Scripts\uvicorn.exe") {
    & ".\venv\Scripts\uvicorn.exe" app:app --reload --port 8001
} elseif (Test-Path ".\.venv\Scripts\uvicorn.exe") {
    & ".\.venv\Scripts\uvicorn.exe" app:app --reload --port 8001
} else {
    uvicorn app:app --reload --port 8001
}
