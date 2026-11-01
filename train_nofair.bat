@echo off
REM Baseline WITHOUT fairness mitigation, for the with-vs-without fairness comparison.
REM Same recipe and seed as model 1 (train_emotion.bat), except:
REM   --class_weight_power 0     no class weights (every image counts the same, majority classes dominate)
REM   --select_metric accuracy   best epoch chosen by validation accuracy instead of macro-F1
REM Then runs the fairness audit, which compares this baseline with the fairness-aware models.
REM About 40 minutes on an RTX 4060 laptop GPU. Close the app first (it uses GPU memory).
REM If it crashes, run this file again: training resumes from the last epoch.
cd /d "%~dp0"
set PY=venv\Scripts\python.exe
if exist "models\emotion_r50_nofair.done" (echo [skip] baseline already trained) else (
  %PY% train_emotion_resnet50_vggface2.py --data_dir "TestDb/DATASET" --weights "models/resnet50_ft_weight.pkl" --split_seed 42 --seed 42 --class_weight_power 0 --select_metric accuracy --out "models/emotion_r50_nofair.pth" --resume || goto :fail
  echo done> "models\emotion_r50_nofair.done"
)
%PY% fairness.py --data_dir "TestDb/DATASET" || goto :fail
echo.
echo All done. Restart the app (run.bat): /predict and the batch PDF now show with vs without fairness.
goto :eof
:fail
echo [error] a step failed - run train_nofair.bat again to resume.
