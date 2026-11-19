import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from risklens_engine import (
    analyze_crypto_risk,
    scan_sensitive_input,
    stream_copilot_response,
)

router = APIRouter(prefix="/api", tags=["RiskLens Copilot"])


class RiskRequest(BaseModel):
    chain: str = "ethereum"
    address: Optional[str] = None
    mode: str = "token"
    contractCode: Optional[str] = None


class ChatStreamRequest(BaseModel):
    reportContext: dict
    query: str


@router.post("/v1/risk")
async def api_risk_audit(req: RiskRequest):
    leak = scan_sensitive_input(req.address or req.contractCode or "")
    if leak:
        raise HTTPException(status_code=400, detail=leak)

    target = req.contractCode if req.mode == "contract" else (req.address or "")
    if not target:
        raise HTTPException(status_code=400, detail="Target address or contract code required.")

    return await analyze_crypto_risk(
        mode=req.mode, chain=req.chain, target=target, contract_code=req.contractCode
    )


@router.post("/chat/stream")
async def api_chat_stream(req: ChatStreamRequest):
    leak = scan_sensitive_input(req.query)
    if leak:
        raise HTTPException(status_code=400, detail=leak)

    return StreamingResponse(
        stream_copilot_response(req.reportContext, req.query),
        media_type="text/event-stream",
    )
