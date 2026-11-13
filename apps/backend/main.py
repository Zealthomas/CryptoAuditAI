import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from web3 import Web3

from config import settings
from database import AsyncSessionLocal, encrypt_api_key, get_db, init_db, test_connection
from models import Alert, AlertResponse, AuditReport, AuditReportResponse, AuditRequest, Exchange, ExchangeCreate, IngestRequest, StatsResponse, Transaction, TransactionResponse, Wallet, WalletCreate
from reports import generate_audit_report, get_user_reports
from retrieval import router as retrieval_router
from risklens_engine import analyze_crypto_risk, scan_sensitive_input, stream_copilot_response

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CryptoAuditAI')

app = FastAPI(title='CryptoAuditAI', description='AI-Native Crypto Auditing Platform - OPTIMIZED', version='2.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://127.0.0.1:5500', 'http://localhost:5500', 'http://127.0.0.1:3000', 'http://localhost:3000', '*'],
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
    allow_headers=['*'],
    expose_headers=['*'],
    max_age=600,
)
app.include_router(retrieval_router)
app.mount('/reports', StaticFiles(directory=settings.REPORTS_DIR), name='reports')

w3 = None
if settings.INFURA_PROJECT_ID:
    try:
        infura_url = f'https://mainnet.infura.io/v3/{settings.INFURA_PROJECT_ID}'
        w3 = Web3(Web3.HTTPProvider(infura_url))
        logger.info(f'Connected to Ethereum via Infura: {w3.is_connected()}')
    except Exception as exc:
        logger.error(f'Failed to connect to Infura: {exc}')


class RiskRequest(BaseModel):
    chain: str = 'ethereum'
    address: Optional[str] = None
    mode: str = 'token'
    contractCode: Optional[str] = None


class ChatStreamRequest(BaseModel):
    reportContext: dict = {}
    query: str = ''


@app.post('/api/v1/risk')
async def api_risk_audit(req: RiskRequest):
    leak = scan_sensitive_input(req.address or req.contractCode or '')
    if leak:
        raise HTTPException(status_code=400, detail=leak)

    target = req.contractCode if req.mode == 'contract' else (req.address or '')
    if not target:
        raise HTTPException(status_code=400, detail='Target address or contract code required.')

    return await analyze_crypto_risk(mode=req.mode, chain=req.chain, target=target, contract_code=req.contractCode)


@app.post('/api/chat/stream')
async def api_chat_stream(req: ChatStreamRequest):
    leak = scan_sensitive_input(req.query)
    if leak:
        raise HTTPException(status_code=400, detail=leak)

    return StreamingResponse(stream_copilot_response(req.reportContext or {}, req.query), media_type='text/event-stream')


@app.get('/health')
async def health_check():
    db_status = await test_connection()
    return {
        'status': 'healthy',
        'version': '2.0.0 - OPTIMIZED',
        'database': 'connected' if db_status else 'disconnected',
        'ollama_configured': bool(settings.OLLAMA_HOST),
        'web3_connected': w3.is_connected() if w3 else False,
        'timestamp': datetime.now().isoformat(),
    }


@app.post('/report', response_model=AuditReportResponse)
async def generate_report(request: AuditRequest, db: AsyncSession = Depends(get_db)):
    try:
        report_data = await generate_audit_report(
            user_id=request.user_id,
            report_type=request.report_type,
            date_from=request.date_from,
            date_to=request.date_to,
        )
        return AuditReportResponse(
            id=report_data['report_id'],
            title=f"{request.report_type.title()} Audit Report",
            report_type=request.report_type,
            summary=report_data['summary'][:500],
            overall_risk_score=report_data['overall_risk_score'],
            compliance_score=report_data['compliance_score'],
            total_transactions=report_data['total_transactions'],
            flagged_transactions=report_data['flagged_transactions'],
            created_at=datetime.now(),
            pdf_path=report_data['pdf_path'],
            json_path=report_data['json_path'],
        )
    except Exception as exc:
        logger.error(f'Report generation failed: {exc}')
        raise HTTPException(status_code=500, detail=f'Report generation failed: {str(exc)}')


@app.post('/chat')
async def chat_endpoint(request: dict):
    query = request.get('query') or request.get('message', '')
    user_id = request.get('user_id', 1)
    if not query:
        return {'response': "I didn't receive a message. How can I help you?"}

    simple_responses = {
        'hello': "Hi! I'm your CryptoAuditAI assistant. How can I help you today?",
        'hi': 'Hello! What would you like to know about your crypto audit?',
        'help': 'I can help you understand your transaction analysis, risk scores, and compliance status.',
        'status': 'Your backend is running correctly! All systems are operational.',
        'test': 'Connection test successful! I\'m ready to assist you.',
    }

    q = query.lower().strip()
    for key, response in simple_responses.items():
        if key in q:
            return {'response': response}

    if hasattr(settings, 'OLLAMA_HOST') and settings.OLLAMA_HOST:
        try:
            context = ''
            async with AsyncSessionLocal() as db:
                stats_result = await db.execute(select(func.count(Transaction.id)).where(Transaction.user_id == user_id))
                total_transactions = stats_result.scalar() or 0
                risk_result = await db.execute(select(func.count(Transaction.id)).where(Transaction.user_id == user_id).where(Transaction.risk_level.in_(['high', 'critical'])))
                high_risk = risk_result.scalar() or 0
                context = f'User has {total_transactions} total transactions, {high_risk} high-risk transactions. '

            prompt = f'''You are a professional crypto audit assistant helping users understand their blockchain transaction risks and compliance.

{context}

User question: {query}

Provide a helpful, professional response about crypto auditing, risk analysis, or compliance. Keep it concise and relevant to cryptocurrency auditing.'''

            async with httpx.AsyncClient() as client:
                ollama_response = await client.post(
                    f'{settings.OLLAMA_HOST}/api/generate',
                    json={'model': getattr(settings, 'OLLAMA_MODEL', 'mistral:instruct'), 'prompt': prompt, 'stream': False, 'options': {'temperature': 0.7, 'top_p': 0.9, 'max_tokens': 200}},
                    timeout=30.0,
                )
                if ollama_response.status_code == 200:
                    result = ollama_response.json()
                    ai_response = result.get('response', '').strip()
                    if ai_response:
                        return {'response': ai_response}
        except Exception as exc:
            logger.error(f'Ollama request failed: {exc}')

    return {'response': f"I understand you're asking about '{query}'. As your crypto audit assistant, I can help analyze transaction risks, compliance issues, and security patterns."}


@app.on_event('startup')
async def startup_event():
    logger.info('Starting CryptoAuditAI backend...')
    await init_db()


if __name__ == '__main__':
    uvicorn.run('apps.backend.main:app', host='0.0.0.0', port=8000, reload=settings.DEBUG, log_level='info')
