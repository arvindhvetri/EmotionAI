# Tidies the repository for GitHub. Run from D:\Research4 in PowerShell AFTER training has finished:
#     powershell -ExecutionPolicy Bypass -File .\cleanup_repo.ps1
# Nothing is deleted: residual files are MOVED to _archive\ (and private notes to _private\); both are git-ignored.
# The script stages the changes with git but does NOT commit or push.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$training = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "train_emotion_resnet50_vggface2|ensemble_emotion|train_rafau" }
if ($training) { Write-Host "A training script is still running - wait for it to finish, then run this again." -ForegroundColor Red; exit 1 }

function Move-Safe($src, $dstDir) {
    if (-not (Test-Path $src)) { return }
    New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
    $name = Split-Path $src -Leaf
    $dst = Join-Path $dstDir $name
    if (Test-Path $dst) { $dst = Join-Path $dstDir ("{0}_{1}" -f (Get-Date -Format "yyyyMMddHHmmss"), $name) }
    Move-Item -LiteralPath $src -Destination $dst
    Write-Host ("  moved {0,-45} -> {1}" -f $src, $dstDir)
}

Write-Host "1) Notebooks -> notebooks\" -ForegroundColor Cyan
Move-Safe "FA_EFER_ResNet18_VGGFace2.ipynb"        "notebooks"
Move-Safe "FA_EFER_ResNet50_VGGFace2_Colab.ipynb"  "notebooks"

Write-Host "2) Earlier experiments -> legacy\" -ForegroundColor Cyan
Move-Safe "notebook_code.py"                   "legacy"
Move-Safe "train_emotion_model.py"             "legacy"
Move-Safe "train_resnet50.py"                  "legacy"
Move-Safe "scratch\train_balanced_emotion.py"  "legacy"
Move-Safe "scratch\fine_tune_more.py"          "legacy"
Move-Safe "scratch\build_rafau_notebook.py"    "legacy"
if (Test-Path "legacy_README.md") { Move-Item -Force "legacy_README.md" "legacy\README.md" }

Write-Host "3) Private notes -> _private\ (contains e-mails and dataset download links - never publish)" -ForegroundColor Cyan
Move-Safe "Linksfordb.txt" "_private"

Write-Host "4) Residual / temporary files -> _archive\" -ForegroundColor Cyan
foreach ($f in @("FIX11.md", "read_ppt.py", "app_integration.py", "app_backup_before_resnet50.py",
                 "TestDb_Batch_Report.pdf", "scratch")) { Move-Safe $f "_archive" }
Get-ChildItem "models" -File | Where-Object { $_.Name -like "*_last.pth" -or $_.Name -like "*.done" } |
    ForEach-Object { Move-Safe $_.FullName "_archive\models" }
Move-Safe "models\emotion_r50_s2024_cw03.pth" "_archive\models"   # interrupted first run of model 3 (replaced by _v2)
Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch "\\venv\\" } | Remove-Item -Recurse -Force

Write-Host "5) Stop tracking datasets and unused files in git" -ForegroundColor Cyan
git rm -r --cached --quiet --ignore-unmatch TestDb RAFAU_label.txt Linksfordb.txt FIX11.md read_ppt.py scratch `
    "models/best_resnet50_raf_au.pth" "models/best_vggface2_emotion.pth" | Out-Null
git add -A
Write-Host ""
Write-Host "Staged changes (first 60):" -ForegroundColor Cyan
git status --short | Select-Object -First 60
Write-Host ""
Write-Host "Check the list, then commit:  git commit -m `"v2.0: ResNet-50 + VGGFace2 emotion engine, honest evaluation, docs, cleanup`"" -ForegroundColor Green
Write-Host "IMPORTANT: the dataset is still in the PUBLIC history of GitHub - see the purge steps before pushing." -ForegroundColor Yellow