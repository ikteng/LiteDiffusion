from __future__ import annotations

from flask import Blueprint, abort, jsonify, request

from .. import store

bp = Blueprint("history", __name__)


@bp.get("/history")
def list_history():
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    items, total = store.list_history(limit=limit, offset=offset)
    return jsonify({
        "items": [item.model_dump(mode="json") for item in items],
        "total": total,
    })


@bp.delete("/history/<item_id>")
def delete_history(item_id: str):
    removed = store.delete_history(item_id)
    if not removed:
        abort(404, description="History item not found")
    return "", 204
