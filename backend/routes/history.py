from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from .. import store
from ..models import HistoryListResponse

router = APIRouter()


@router.get("/history", response_model=HistoryListResponse)
def list_history(limit: int = 50, offset: int = 0) -> HistoryListResponse:
    items, total = store.list_history(limit=limit, offset=offset)
    return HistoryListResponse(items=items, total=total)


@router.delete("/history/{item_id}", status_code=204)
def delete_history(item_id: str) -> Response:
    removed = store.delete_history(item_id)
    if not removed:
        raise HTTPException(status_code=404, detail="History item not found")
    return Response(status_code=204)
