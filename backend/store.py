"""Disk persistence for generation history: outputs/index.json + image files."""

from __future__ import annotations

import json
import threading

from . import config
from .models import HistoryItem

_lock = threading.Lock()


def _read_index() -> list[dict]:
    if not config.INDEX_PATH.exists():
        return []
    with config.INDEX_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", [])


def _write_index(items: list[dict]) -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with config.INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump({"items": items}, f, indent=2)


def _to_file_url(file: str) -> str:
    return f"/outputs/{file}"


def append_history(item: HistoryItem) -> None:
    with _lock:
        items = _read_index()
        items.append(item.model_dump(mode="json", exclude={"file_url"}))
        _write_index(items)


def list_history(limit: int = 50, offset: int = 0) -> tuple[list[HistoryItem], int]:
    with _lock:
        items = _read_index()
    items.sort(key=lambda r: r["created_at"], reverse=True)
    total = len(items)
    page = items[offset : offset + limit]
    return [HistoryItem(**r, file_url=_to_file_url(r["file"])) for r in page], total


def delete_history(item_id: str) -> bool:
    with _lock:
        items = _read_index()
        remaining = [r for r in items if r["id"] != item_id]
        removed = [r for r in items if r["id"] == item_id]
        if not removed:
            return False
        _write_index(remaining)

    image_path = config.IMAGES_DIR / removed[0]["file"].split("/")[-1]
    image_path.unlink(missing_ok=True)
    return True
