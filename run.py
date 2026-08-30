"""Single entrypoint for running LiteDiffusion locally: builds the API + serves the built frontend."""

from __future__ import annotations

import os
import webbrowser

import uvicorn

from backend import config

HOST = os.environ.get("LITEDIFFUSION_HOST", "127.0.0.1")
PORT = int(os.environ.get("LITEDIFFUSION_PORT", "8000"))


def main() -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if not config.FRONTEND_DIST_DIR.exists():
        print(
            "frontend/dist not found — build the frontend first:\n"
            "  cd frontend\n"
            "  npm install\n"
            "  npm run build\n"
            "Then re-run: python run.py"
        )
        return

    if os.environ.get("LITEDIFFUSION_NO_BROWSER") != "1":
        try:
            webbrowser.open(f"http://{HOST}:{PORT}")
        except Exception:
            pass
    uvicorn.run("backend.main:app", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
