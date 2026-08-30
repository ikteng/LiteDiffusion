@echo off
cd /d "%~dp0"

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

python -c "import fastapi, imageio, diffusers, torch, PIL, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing Python dependencies...
    pip install -r requirements.txt
)

if not exist "frontend\dist" (
    echo Building frontend...
    pushd frontend
    call npm install
    call npm run build
    popd
)

echo Starting LiteDiffusion...
python run.py
pause
