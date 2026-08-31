@echo off
cd /d "%~dp0"

if "%1"=="dev" goto dev

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
goto end

:dev
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

if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    pushd frontend
    call npm install
    popd
)

echo Starting backend with auto-reload...
start "LiteDiffusion Backend" cmd /c "call .venv\Scripts\activate.bat && uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"

echo Starting frontend dev server...
start "LiteDiffusion Frontend" cmd /c "cd /d %~dp0frontend && npm run dev"

timeout /t 3 /nobreak >nul
echo Opening browser...
start http://127.0.0.1:5173

echo.
echo Both windows should open. Close them to stop the dev servers.
pause

:end
