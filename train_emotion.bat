@echo off
REM Trains the ResNet-50 + VGGFace2 emotion model using the project venv (CUDA build of PyTorch).
REM Extra options are passed through, e.g.  train_emotion.bat --resume   or   train_emotion.bat --eval_only
cd /d "%~dp0"
"venv\Scripts\python.exe" train_emotion_resnet50_vggface2.py --data_dir "TestDb/DATASET" --weights "models/resnet50_ft_weight.pkl" --out "models/best_resnet50_vggface2_emotion.pth" %*
