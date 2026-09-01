#!/bin/bash
set -e

cd "$(dirname "$0")"

# Check for the existence of a virtual environment folder
if [ -d ".venv" ]; then
  echo "Virtual environment already exists."
else
  echo "Creating virtual environment..."
  if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    python3 -m venv .venv
  else
    python -m venv .venv
  fi
  echo "Virtual environment created."
fi

echo "Activating virtual environment..."
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
  source .venv/bin/activate
else
  source .venv/Scripts/activate
fi

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! Virtual environment is active and dependencies are installed."

# Kill whatever is listening on a port, walking each match's full process
# tree (//T). Used both to reclaim ports left by a previous session that
# didn't shut down cleanly, and to tear down our own backend/frontend on
# exit — plain `kill`/`kill 0` only reaches the immediate shell here, not
# the Flask --debug reloader's child or Vite's node process, since Git
# Bash on Windows doesn't map process groups to real Win32 process trees.
free_port() {
  local port="$1"
  if command -v netstat >/dev/null 2>&1; then
    for pid in $(netstat -ano -p tcp 2>/dev/null | awk -v p=":$port" '$2 ~ p && $4 == "LISTENING" {print $5}' | sort -u); do
      [ -n "$pid" ] && [ "$pid" != "0" ] && taskkill //F //T //PID "$pid" >/dev/null 2>&1 || true
    done
  fi
}

cleanup() {
  free_port 8000
  free_port 5173
}
trap cleanup EXIT INT TERM

free_port 8000
free_port 5173

# Run the Flask backend in the background
echo "Running backend..."
export FLASK_APP=backend.main:app
python -m flask run --host 127.0.0.1 --port 8000 --debug &

# Navigate to the frontend directory and start Vite
echo "Running frontend..."
cd frontend || { echo "Frontend directory not found!"; exit 1; }
npm install
npm run dev &
cd ..

# Wait for both processes to finish
wait
