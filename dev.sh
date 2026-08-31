#!/usr/bin/env bash
# Dev mode: auto-reload backend + Vite hot-reload frontend.
# Run: ./dev.sh
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment..."
  python -m venv .venv
fi

if [[ -f ".venv/Scripts/activate" ]]; then
  source .venv/Scripts/activate
else
  source .venv/bin/activate
fi

if [[ -x ".venv/Scripts/python.exe" ]]; then
  PY=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

$PY -c "import fastapi, imageio, diffusers, torch, PIL, numpy" >/dev/null 2>&1 || {
  echo "Installing Python dependencies..."
  pip install -r requirements.txt
}

if [[ ! -d "frontend/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  pushd frontend >/dev/null
  npm install
  popd >/dev/null
fi

echo "Starting backend with auto-reload..."
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "Starting frontend dev server..."
pushd frontend >/dev/null
npm run dev &
FRONTEND_PID=$!
popd >/dev/null

sleep 5
echo "Opening browser..."
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open http://127.0.0.1:5173 >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
  open http://127.0.0.1:5173 >/dev/null 2>&1 || true
elif command -v cmd.exe >/dev/null 2>&1; then
  cmd.exe /c start http://127.0.0.1:5173 >/dev/null 2>&1 || true
fi

echo
echo "Dev servers started. Close the terminal to stop."
wait $BACKEND_PID $FRONTEND_PID
