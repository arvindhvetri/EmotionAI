@echo off
echo ========================================================
echo Starting Facial Action Unit & Emotion Recognition App...
echo ========================================================
echo URL: http://127.0.0.1:8001
echo Press Ctrl+C in this terminal to stop the server.
echo ========================================================
if exist ".\venv\Scripts\uvicorn.exe" (
    .\venv\Scripts\uvicorn.exe app:app --reload --port 8001
) else if exist ".\.venv\Scripts\uvicorn.exe" (
    .\.venv\Scripts\uvicorn.exe app:app --reload --port 8001
) else (
    uvicorn app:app --reload --port 8001
)
pause
