@echo off
REM Train emotion models 2 and 3 (class-weight power 0.3) and build the validation-selected ensemble.
REM Model 1 (models\best_resnet50_vggface2_emotion.pth) is trained separately with train_emotion.bat.
REM If a run crashes, start this file again: finished models are skipped and an interrupted one resumes.
cd /d "%~dp0"
set PY=venv\Scripts\python.exe
set COMMON=--data_dir "TestDb/DATASET" --weights "models/resnet50_ft_weight.pkl" --split_seed 42 --class_weight_power 0.3

if exist "models\emotion_r50_s7_cw03.done" (echo [skip] model 2 already trained) else (
  %PY% train_emotion_resnet50_vggface2.py %COMMON% --seed 7 --out "models/emotion_r50_s7_cw03.pth" --resume || goto :fail
  echo done> "models\emotion_r50_s7_cw03.done"
)
if exist "models\emotion_r50_s2024_cw03_v2.done" (echo [skip] model 3 already trained) else (
  %PY% train_emotion_resnet50_vggface2.py %COMMON% --seed 2024 --patience 8 --out "models/emotion_r50_s2024_cw03_v2.pth" --resume || goto :fail
  echo done> "models\emotion_r50_s2024_cw03_v2.done"
)

%PY% ensemble_emotion.py --data_dir "TestDb/DATASET" --models "models/best_resnet50_vggface2_emotion.pth" "models/emotion_r50_s7_cw03.pth" "models/emotion_r50_s2024_cw03_v2.pth" || goto :fail
echo.
echo All done. Restart the app (run.bat) to use models\emotion_ensemble.json
goto :eof
:fail
echo [error] a step failed - run train_more.bat again to resume.
