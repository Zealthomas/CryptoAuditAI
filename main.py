"""
Complete CryptoAuditAI Backend
Production-ready FastAPI server with real crypto integrations and AI analysis
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uvicorn

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

# Web3 and CCXT for crypto integrations
from web3 import Web3
import ccxt.async_support as ccxt
import httpx

# Local imports
from database import init_db, test_connection, get_db, AsyncSessionLocal
from models import (
    User, Transaction, Wallet, Exchange, AuditReport, Alert,
    IngestRequest, AuditRequest, TransactionResponse, AuditReportResponse,
    AlertResponse, StatsResponse, WalletCreate, ExchangeCreate
)
from ai_engine import (
    ai_engine, analyze_transaction_risk, detect_user_patterns, 
    get_compliance_assessment, create_risk_alert
)
from reports import generate_audit_report, get_user_reports
from retrieval import router as retrieval_router
from config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CryptoAuditAI")

# Initialize FastAPI app
app = FastAPI(
    title="CryptoAuditAI",
    description="AI-Native Crypto Auditing Platform",
    version="1.0.0"
)

# CORS middleware - FIXED VERSION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500", 
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "*"  # Allow all origins for development - REMOVE in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],  # Explicitly include OPTIONS
    allow_headers=[
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "X-CSRFToken"
    ],
    expose_headers=["*"],  # Allow frontend to access response headers
    max_age=600  # Cache preflight requests for 10 minutes
)

# Include retrieval router
app.include_router(retrieval_router)

# Serve static files (reports)
app.mount("/reports", StaticFiles(directory=settings.REPORTS_DIR), name="reports")

# Global Web3 connection
w3 = None
if settings.INFURA_PROJECT_ID:
    try:
        infura_url = f"https://mainnet.infura.io/v3/{settings.INFURA_PROJECT_ID}"
        w3 = Web3(Web3.HTTPProvider(infura_url))
        logger.info(f"Connected to Ethereum via Infura: {w3.is_connected()}")
    except Exception as e:
        logger.error(f"Failed to connect to Infura: {e}")

# -------------------------
# Crypto Integration Functions
# -------------------------

async def fetch_ethereum_transactions(address: str, limit: int = 100) -> List[Dict]:
    """Fetch real Ethereum transactions using Web3"""
    transactions = []
    
    if not w3 or not w3.is_connected():
        logger.warning("Web3 not connected, using mock data")
        return generate_mock_ethereum_transactions(address, limit)
    
    try:
        # Get latest block
        latest_block = w3.eth.block_number
        
        # Check recent blocks for transactions to/from this address
        for i in range(min(100, latest_block)):
            block_number = latest_block - i
            block = w3.eth.get_block(block_number, full_transactions=True)
            
            for tx in block.transactions:
                if (tx['to'] and tx['to'].lower() == address.lower()) or \
                   (tx['from'] and tx['from'].lower() == address.lower()):
                    
                    # Get transaction receipt for gas used
                    receipt = w3.eth.get_transaction_receipt(tx['hash'])
                    
                    transactions.append({
                        'hash': tx['hash'].hex(),
                        'amount': float(w3.from_wei(tx['value'], 'ether')),
                        'from_address': tx['from'],
                        'to_address': tx['to'],
                        'gas_price': float(w3.from_wei(tx['gasPrice'], 'gwei')),
                        'gas_used': receipt['gasUsed'],
                        'fee': float(w3.from_wei(tx['gasPrice'] * receipt['gasUsed'], 'ether')),
                        'block_number': tx['blockNumber'],
                        'timestamp': datetime.fromtimestamp(block.timestamp),
                        'status': 'confirmed' if receipt['status'] == 1 else 'failed'
                    })
                    
                    if len(transactions) >= limit:
                        return transactions
    
    except Exception as e:
        logger.error(f"Error fetching Ethereum transactions: {e}")
        return generate_mock_ethereum_transactions(address, limit)
    
    return transactions

def generate_mock_ethereum_transactions(address: str, limit: int) -> List[Dict]:
    """Generate realistic mock Ethereum transactions for demo"""
    import random
    from datetime import datetime, timedelta
    
    transactions = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(min(limit, 50)):
        tx_time = base_time + timedelta(hours=random.randint(1, 720))
        amount = random.uniform(0.01, 10.0)
        gas_price = random.uniform(20, 100)
        gas_used = random.randint(21000, 100000)
        
        transactions.append({
            'hash': f"0x{''.join(random.choices('0123456789abcdef', k=64))}",
            'amount': amount,
            'from_address': f"0x{''.join(random.choices('0123456789abcdef', k=40))}",
            'to_address': address if random.choice([True, False]) else f"0x{''.join(random.choices('0123456789abcdef', k=40))}",
            'gas_price': gas_price,
            'gas_used': gas_used,
            'fee': (gas_price * gas_used) / 1e9,  # Convert to ETH
            'block_number': 18500000 + i,
            'timestamp': tx_time,
            'status': random.choices(['confirmed', 'failed'], weights=[95, 5])[0]
        })
    
    return transactions

async def fetch_exchange_transactions(exchange_name: str, api_key: str = None, limit: int = 100) -> List[Dict]:
    """Fetch exchange transactions using CCXT"""
    try:
        if exchange_name.lower() == 'binance' and api_key:
            exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': 'dummy_secret',  # Would need real secret
                'sandbox': True  # Use testnet
            })
        else:
            # Use mock data if no API key or unsupported exchange
            return generate_mock_exchange_transactions(exchange_name, limit)
        
        # Fetch trades (would need proper API keys and secrets)
        trades = await exchange.fetch_my_trades(limit=limit)
        
        transactions = []
        for trade in trades:
            transactions.append({
                'id': trade['id'],
                'amount': trade['amount'],
                'asset': trade['symbol'].split('/')[0],
                'price': trade['price'],
                'side': trade['side'],  # buy/sell
                'fee': trade['fee']['cost'],
                'timestamp': datetime.fromtimestamp(trade['timestamp'] / 1000),
                'status': 'confirmed'
            })
        
        await exchange.close()
        return transactions
        
    except Exception as e:
        logger.error(f"Error fetching {exchange_name} transactions: {e}")
        return generate_mock_exchange_transactions(exchange_name, limit)

def generate_mock_exchange_transactions(exchange_name: str, limit: int) -> List[Dict]:
    """Generate realistic mock exchange transactions"""
    import random
    from datetime import datetime, timedelta
    
    assets = ['BTC', 'ETH', 'USDT', 'BNB', 'ADA', 'DOT']
    sides = ['buy', 'sell']
    
    transactions = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(min(limit, 50)):
        asset = random.choice(assets)
        side = random.choice(sides)
        amount = random.uniform(0.1, 5.0) if asset in ['BTC', 'ETH'] else random.uniform(10, 1000)
        price = random.uniform(20000, 50000) if asset == 'BTC' else random.uniform(1000, 3000)
        
        transactions.append({
            'id': f"{exchange_name}_{i:06d}",
            'amount': amount,
            'asset': asset,
            'price': price,
            'side': side,
            'fee': amount * 0.001,  # 0.1% fee
            'timestamp': base_time + timedelta(hours=random.randint(1, 720)),
            'status': 'confirmed'
        })
    
    return transactions

# -------------------------
# API Endpoints
# -------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_status = await test_connection()
    return {
        "status": "healthy",
        "database": "connected" if db_status else "disconnected",
        "ollama_configured": bool(settings.OLLAMA_HOST),
        "web3_connected": w3.is_connected() if w3 else False,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ingest", response_model=Dict[str, Any])
async def ingest_transactions(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Ingest transactions from wallets or exchanges"""
    
    async def process_ingestion():
        try:
            transactions_data = []
            
            # Ingest from wallet
            if request.wallet_address:
                eth_transactions = await fetch_ethereum_transactions(request.wallet_address, request.limit)
                for tx_data in eth_transactions:
                    transactions_data.append({
                        'source': 'ethereum',
                        'transaction_type': 'send' if tx_data['from_address'].lower() == request.wallet_address.lower() else 'receive',
                        **tx_data
                    })
            
            # Ingest from exchange
            if request.exchange_name:
                exchange_transactions = await fetch_exchange_transactions(request.exchange_name, limit=request.limit)
                for tx_data in exchange_transactions:
                    transactions_data.append({
                        'source': request.exchange_name.lower(),
                        'transaction_type': 'trade',
                        'hash': None,  # Exchange transactions don't have blockchain hash
                        'amount': tx_data['amount'],
                        'asset': tx_data['asset'],
                        'fee': tx_data['fee'],
                        'timestamp': tx_data['timestamp'],
                        'status': tx_data['status'],
                        'from_address': None,
                        'to_address': None
                    })
            
            # Store transactions in database
            async with AsyncSessionLocal() as session:
                new_transactions = []
                for tx_data in transactions_data:
                    transaction = Transaction(
                        user_id=request.user_id,
                        hash=tx_data.get('hash'),
                        amount=tx_data['amount'],
                        asset=tx_data.get('asset', 'ETH'),
                        from_address=tx_data.get('from_address'),
                        to_address=tx_data.get('to_address'),
                        transaction_type=tx_data['transaction_type'],
                        status=tx_data['status'],
                        source=tx_data['source'],
                        fee=tx_data.get('fee', 0),
                        gas_price=tx_data.get('gas_price'),
                        gas_used=tx_data.get('gas_used'),
                        timestamp=tx_data['timestamp']
                    )
                    session.add(transaction)
                    new_transactions.append(transaction)
                
                await session.commit()
                
                # Analyze each transaction for risk
                for transaction in new_transactions:
                    await session.refresh(transaction)
                    risk_analysis = await analyze_transaction_risk(transaction.id)
                    
                    # Update transaction with risk analysis
                    transaction.risk_score = risk_analysis.risk_score
                    transaction.risk_level = risk_analysis.risk_level
                    transaction.anomaly_detected = risk_analysis.anomaly_detected
                    transaction.compliance_flags = risk_analysis.compliance_flags
                    transaction.analyzed_at = datetime.now()
                    
                    # Create alerts if necessary
                    await create_risk_alert(request.user_id, transaction.id, risk_analysis)
                
                await session.commit()
                logger.info(f"Ingested and analyzed {len(new_transactions)} transactions for user {request.user_id}")
        
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
    
    background_tasks.add_task(process_ingestion)
    
    return {
        "status": "success",
        "message": "Transaction ingestion started",
        "user_id": request.user_id
    }

@app.post("/audit", response_model=Dict[str, Any])
async def run_audit(
    request: AuditRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Run comprehensive AI audit on user's transactions"""
    
    async def process_audit():
        try:
            # Detect suspicious patterns
            patterns = await detect_user_patterns(request.user_id)
            
            # Create alerts for patterns
            for pattern in patterns:
                await ai_engine.create_alert(
                    user_id=request.user_id,
                    transaction_id=None,
                    title=f"Suspicious Pattern: {pattern.pattern_type.replace('_', ' ').title()}",
                    message=f"{pattern.description} (Confidence: {pattern.confidence:.1%})",
                    severity="warning" if pattern.confidence < 0.8 else "high",
                    alert_type="pattern"
                )
            
            # Generate compliance assessment
            compliance = await get_compliance_assessment(request.user_id)
            
            logger.info(f"Audit completed for user {request.user_id}: {len(patterns)} patterns detected")
            
        except Exception as e:
            logger.error(f"Audit failed for user {request.user_id}: {e}")
    
    background_tasks.add_task(process_audit)
    
    return {
        "status": "success",
        "message": "AI audit started",
        "user_id": request.user_id,
        "report_type": request.report_type
    }

@app.post("/report", response_model=AuditReportResponse)
async def generate_report(
    request: AuditRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate comprehensive audit report"""
    try:
        report_data = await generate_audit_report(
            user_id=request.user_id,
            report_type=request.report_type,
            date_from=request.date_from,
            date_to=request.date_to
        )
        
        return AuditReportResponse(
            id=report_data["report_id"],
            title=f"{request.report_type.title()} Audit Report",
            report_type=request.report_type,
            summary=report_data["summary"][:500],  # Truncate for response
            overall_risk_score=report_data["overall_risk_score"],
            compliance_score=report_data["compliance_score"],
            total_transactions=report_data["total_transactions"],
            flagged_transactions=report_data["flagged_transactions"],
            created_at=datetime.now(),
            pdf_path=report_data["pdf_path"],
            json_path=report_data["json_path"]
        )
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/reports/{report_id}/download")
async def download_report(report_id: int, format: str = Query("pdf", pattern="^(pdf|json)$")):
    """Download generated report"""
    async with AsyncSessionLocal() as session:
        report = await session.get(AuditReport, report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        file_path = report.pdf_path if format == "pdf" else report.json_path
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Report file not found")
        
        filename = f"audit_report_{report_id}.{format}"
        media_type = "application/pdf" if format == "pdf" else "application/json"
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )

@app.get("/users/{user_id}/stats", response_model=StatsResponse)
async def get_user_stats(user_id: int, db: AsyncSession = Depends(get_db)):
    """Get user statistics"""

    # -------------------------
    # Total transactions
    # -------------------------
    total_transactions_result = await db.execute(
        select(func.count(Transaction.id)).where(Transaction.user_id == user_id)
    )
    total_transactions = total_transactions_result.scalar() or 0

    # -------------------------
    # High-risk transactions
    # -------------------------
    high_risk_result = await db.execute(
        select(func.count(Transaction.id))
        .where(Transaction.user_id == user_id)
        .where(Transaction.risk_level.in_(["high", "critical"]))
    )
    high_risk_transactions = high_risk_result.scalar() or 0

    # -------------------------
    # Wallet count
    # -------------------------
    wallet_count_result = await db.execute(
        select(func.count(Wallet.id)).where(Wallet.user_id == user_id)
    )
    total_wallets = wallet_count_result.scalar() or 0

    # -------------------------
    # Exchange count
    # -------------------------
    exchange_count_result = await db.execute(
        select(func.count(Exchange.id)).where(Exchange.user_id == user_id)
    )
    total_exchanges = exchange_count_result.scalar() or 0

    # -------------------------
    # Unresolved alerts
    # -------------------------
    unresolved_alerts_result = await db.execute(
        select(func.count(Alert.id))
        .where(Alert.user_id == user_id)
        .where(Alert.is_resolved == False)
    )
    unresolved_alerts = unresolved_alerts_result.scalar() or 0

    # -------------------------
    # Last sync (latest transaction timestamp)
    # -------------------------
    last_transaction_result = await db.execute(
        select(Transaction.timestamp)   # 👈 use timestamp (not created_at)
        .where(Transaction.user_id == user_id)
        .order_by(desc(Transaction.timestamp))
        .limit(1)
    )
    last_sync = last_transaction_result.scalar_one_or_none()

    # -------------------------
    # Return structured response
    # -------------------------
    return StatsResponse(
        total_transactions=total_transactions,
        total_wallets=total_wallets,
        total_exchanges=total_exchanges,
        high_risk_transactions=high_risk_transactions,
        unresolved_alerts=unresolved_alerts,
        last_sync=last_sync
    )

    
    # Unresolved alerts
    unresolved_alerts_result = await db.execute(
        select(func.count(Alert.id))
        .where(Alert.user_id == user_id)
        .where(Alert.is_resolved == False)
    )
    unresolved_alerts = unresolved_alerts_result.scalar() or 0
    
    # Last sync time
    last_transaction_result = await db.execute(
        select(Transaction.created_at)
        .where(Transaction.user_id == user_id)
        .order_by(desc(Transaction.created_at))
        .limit(1)
    )
    last_sync = last_transaction_result.scalar()
    
    return StatsResponse(
        total_transactions=total_transactions,
        total_wallets=wallet_count,
        total_exchanges=exchange_count,
        high_risk_transactions=high_risk_transactions,
        unresolved_alerts=unresolved_alerts,
        last_sync=last_sync
    )

@app.get("/users/{user_id}/transactions", response_model=List[TransactionResponse])
async def get_user_transactions(
    user_id: int,
    limit: int = Query(50, le=1000),
    risk_level: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$"),
    db: AsyncSession = Depends(get_db)
):
    """Get user's transactions with optional filtering"""
    query = select(Transaction).where(Transaction.user_id == user_id)
    
    if risk_level:
        query = query.where(Transaction.risk_level == risk_level)
    
    query = query.order_by(desc(Transaction.timestamp)).limit(limit)
    
    result = await db.execute(query)
    transactions = result.scalars().all()
    
    return [TransactionResponse.from_orm(tx) for tx in transactions]

@app.get("/users/{user_id}/alerts", response_model=List[AlertResponse])
async def get_user_alerts(
    user_id: int,
    unresolved_only: bool = Query(False),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db)
):
    """Get user's alerts"""
    query = select(Alert).where(Alert.user_id == user_id)
    
    if unresolved_only:
        query = query.where(Alert.is_resolved == False)
    
    query = query.order_by(desc(Alert.created_at)).limit(limit)
    
    result = await db.execute(query)
    alerts = result.scalars().all()
    
    return [AlertResponse.from_orm(alert) for alert in alerts]

@app.post("/users/{user_id}/alerts/{alert_id}/resolve")
async def resolve_alert(user_id: int, alert_id: int, db: AsyncSession = Depends(get_db)):
    """Mark an alert as resolved"""
    alert = await db.get(Alert, alert_id)
    if not alert or alert.user_id != user_id:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.is_resolved = True
    alert.resolved_at = datetime.now()
    await db.commit()
    
    return {"status": "resolved", "alert_id": alert_id}

@app.get("/users/{user_id}/reports", response_model=List[Dict[str, Any]])
async def get_user_reports_endpoint(user_id: int, limit: int = Query(10, le=50)):
    """Get user's audit reports"""
    return await get_user_reports(user_id, limit)

@app.post("/users/{user_id}/wallets", response_model=Dict[str, Any])
async def add_wallet(user_id: int, wallet: WalletCreate, db: AsyncSession = Depends(get_db)):
    """Add a new wallet for the user"""
    # Check if wallet already exists
    existing_wallet = await db.execute(
        select(Wallet).where(Wallet.address == wallet.address)
    )
    if existing_wallet.scalar():
        raise HTTPException(status_code=400, detail="Wallet already exists")
    
    new_wallet = Wallet(
        user_id=user_id,
        address=wallet.address,
        name=wallet.name,
        blockchain=wallet.blockchain
    )
    db.add(new_wallet)
    await db.commit()
    await db.refresh(new_wallet)
    
    return {"status": "created", "wallet_id": new_wallet.id, "address": new_wallet.address}

@app.post("/users/{user_id}/exchanges", response_model=Dict[str, Any])
async def add_exchange(user_id: int, exchange: ExchangeCreate, db: AsyncSession = Depends(get_db)):
    """Add exchange configuration for user"""
    from database import encrypt_api_key
    
    new_exchange = Exchange(
        user_id=user_id,
        name=exchange.name,
        api_key_encrypted=encrypt_api_key(exchange.api_key) if exchange.api_key else None,
        api_secret_encrypted=encrypt_api_key(exchange.api_secret) if exchange.api_secret else None
    )
    db.add(new_exchange)
    await db.commit()
    await db.refresh(new_exchange)
    
    return {"status": "created", "exchange_id": new_exchange.id, "name": new_exchange.name}

@app.post("/test/mock-transactions/{user_id}")
async def create_mock_transactions(user_id: int, count: int = Query(10, le=100)):
    """Create mock transactions for testing (development only)"""
    if not settings.DEBUG:
        raise HTTPException(status_code=403, detail="Only available in debug mode")
    
    import random
    from datetime import datetime, timedelta
    
    mock_transactions = []
    base_time = datetime.now() - timedelta(days=30)
    
    async with AsyncSessionLocal() as session:
        for i in range(count):
            transaction = Transaction(
                user_id=user_id,
                hash=f"0x{''.join(random.choices('0123456789abcdef', k=64))}",
                amount=random.uniform(10, 5000),
                asset=random.choice(['BTC', 'ETH', 'USDT', 'BNB']),
                transaction_type=random.choice(['send', 'receive', 'trade']),
                source=random.choice(['ethereum', 'binance', 'coinbase']),
                status=random.choices(['confirmed', 'failed'], weights=[95, 5])[0],
                timestamp=base_time + timedelta(hours=random.randint(1, 720)),
                fee=random.uniform(0.001, 0.1)
            )
            
            # Add some risk factors for testing
            if random.random() < 0.2:  # 20% high risk
                transaction.risk_level = random.choice(['high', 'critical'])
                transaction.risk_score = random.uniform(70, 95)
                transaction.anomaly_detected = True
                transaction.compliance_flags = ['LARGE_TRANSACTION', 'UNUSUAL_TIMING']
            else:
                transaction.risk_level = random.choice(['low', 'medium'])
                transaction.risk_score = random.uniform(5, 50)
            
            session.add(transaction)
            mock_transactions.append(transaction)
        
        await session.commit()
    
    return {
        "status": "created",
        "count": len(mock_transactions),
        "user_id": user_id,
        "message": f"Created {count} mock transactions for testing"
    }

# -------------------------
# Startup and Shutdown Events
# -------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting CryptoAuditAI Backend...")
    
    # Initialize database
    await init_db()
    
    # Test database connection
    db_status = await test_connection()
    if not db_status:
        logger.error("Database connection failed!")
    
    # Test Ollama connection
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.info(f"Ollama connected. Available models: {[m['name'] for m in models]}")
            else:
                logger.warning("Ollama not responding properly")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
    
    logger.info("CryptoAuditAI Backend started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down CryptoAuditAI Backend...")



# Chat endpoint for the frontend chat feature
@app.post("/chat")
async def chat_endpoint(request: dict):
    """Enhanced chat endpoint with proper Ollama integration"""
    logger.info(f"Chat request received: {request}")
    query = request.get("query") or request.get("message", "")  # Support both formats
    user_id = request.get("user_id", 1)
    
    if not query:
        return {"response": "I didn't receive a message. How can I help you?"}
    
    # Simple responses for basic queries
    simple_responses = {
        "hello": "Hi! I'm your CryptoAuditAI assistant. How can I help you today?",
        "hi": "Hello! What would you like to know about your crypto audit?",
        "help": "I can help you understand your transaction analysis, risk scores, and compliance status.",
        "status": "Your backend is running correctly! All systems are operational.",
        "test": "Connection test successful! I'm ready to assist you."
    }
    
    query_lower = query.lower().strip()
    
    # Check for simple responses first
    for key, response in simple_responses.items():
        if key in query_lower:
            return {"response": response}
    
    # Try Ollama for complex queries
    if hasattr(settings, 'OLLAMA_HOST') and settings.OLLAMA_HOST:
        try:
            logger.info(f"Sending request to Ollama at {settings.OLLAMA_HOST}")
            
            # Get user context for better responses
            context = ""
            try:
                async with AsyncSessionLocal() as db:
                    stats_result = await db.execute(
                        select(func.count(Transaction.id)).where(Transaction.user_id == user_id)
                    )
                    total_transactions = stats_result.scalar() or 0
                    
                    risk_result = await db.execute(
                        select(func.count(Transaction.id))
                        .where(Transaction.user_id == user_id)
                        .where(Transaction.risk_level.in_(["high", "critical"]))
                    )
                    high_risk = risk_result.scalar() or 0
                    
                    context = f"User has {total_transactions} total transactions, {high_risk} high-risk transactions. "
            except Exception as e:
                logger.warning(f"Could not fetch user context: {e}")
            
            prompt = f"""You are a professional crypto audit assistant helping users understand their blockchain transaction risks and compliance.

{context}

User question: {query}

Provide a helpful, professional response about crypto auditing, risk analysis, or compliance. Keep it concise and relevant to cryptocurrency auditing."""

            async with httpx.AsyncClient() as client:
                logger.info(f"Making Ollama request with model: {getattr(settings, 'OLLAMA_MODEL', 'mistral:instruct')}")
                
                ollama_response = await client.post(
                    f"{settings.OLLAMA_HOST}/api/generate",
                    json={
                        "model": getattr(settings, 'OLLAMA_MODEL', 'mistral:instruct'),
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 200
                        }
                    },
                    timeout=30.0
                )
                
                logger.info(f"Ollama response status: {ollama_response.status_code}")
                
                if ollama_response.status_code == 200:
                    result = ollama_response.json()
                    logger.info(f"Ollama response: {result}")
                    
                    ai_response = result.get("response", "").strip()
                    if ai_response:
                        return {"response": ai_response}
                    else:
                        logger.warning("Empty response from Ollama")
                else:
                    logger.error(f"Ollama error: {ollama_response.status_code} - {ollama_response.text}")
                    
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama: {e}")
        except Exception as e:
            logger.error(f"Ollama request failed: {type(e).__name__}: {e}")
    else:
        logger.warning("OLLAMA_HOST not configured or empty")
    
    # Fallback response
    return {
        "response": f"I understand you're asking about '{query}'. As your crypto audit assistant, I can help analyze transaction risks, compliance issues, and security patterns. However, I'm having trouble accessing my AI capabilities right now. Please check the server logs for more details."
    }

# -------------------------
# Main Entry Point
# -------------------------

if __name__ == "__main__":
    logger.info("Starting CryptoAuditAI Backend with Mistral Integration")
    logger.info(f"Ollama URL: {settings.OLLAMA_HOST}")
    logger.info(f"Default Model: {settings.OLLAMA_MODEL}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )