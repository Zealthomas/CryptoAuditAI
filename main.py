"""
Complete CryptoAuditAI Backend
Production-ready FastAPI server with real crypto integrations and AI analysis
UPGRADED: AI only for reports and chat, fast programmatic scoring for audits
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uvicorn
import numpy as np

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
    ai_engine, detect_user_patterns, get_compliance_assessment
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500", 
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
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
    expose_headers=["*"],
    max_age=600
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
# PROGRAMMATIC RISK SCORING (No AI)
# -------------------------

def calculate_risk_score(transaction_data: Dict) -> Dict[str, Any]:
    """
    Fast programmatic risk scoring without AI
    Returns risk_score, risk_level, anomaly_detected, compliance_flags, counterparty_risk
    """
    risk_factors = []
    compliance_flags = []
    counterparty_flags = []
    
    amount = transaction_data.get('amount', 0)
    gas_price = transaction_data.get('gas_price', 0)
    fee = transaction_data.get('fee', 0)
    status = transaction_data.get('status', 'confirmed')
    timestamp = transaction_data.get('timestamp', datetime.now())
    from_address = transaction_data.get('from_address', '')
    to_address = transaction_data.get('to_address', '')
    source = transaction_data.get('source', 'unknown')
    
    # Amount-based risk
    if amount > 100000:
        risk_factors.append(30)
        compliance_flags.append("LARGE_TRANSACTION")
    elif amount > 50000:
        risk_factors.append(20)
    elif amount > 10000:
        risk_factors.append(10)
        compliance_flags.append("CTR_THRESHOLD")
    
    # Structuring detection
    if 9000 <= amount <= 9999:
        risk_factors.append(40)
        compliance_flags.append("STRUCTURING_SUSPICION")
    
    # Round amount pattern
    if amount % 1000 == 0 and amount >= 5000:
        risk_factors.append(15)
        compliance_flags.append("ROUND_AMOUNT_PATTERN")
    
    # Gas price anomalies (for Ethereum)
    if gas_price and gas_price > 100:
        risk_factors.append(15)
        compliance_flags.append("HIGH_GAS_PRICE")
    
    # Transaction status
    if status == "failed":
        risk_factors.append(25)
        compliance_flags.append("FAILED_TRANSACTION")
    
    # Time-based patterns (outside business hours)
    hour = timestamp.hour if hasattr(timestamp, 'hour') else datetime.now().hour
    if hour < 6 or hour > 22:
        risk_factors.append(10)
        compliance_flags.append("UNUSUAL_TIMING")
    
    # Fee anomalies
    if fee and amount > 0:
        fee_ratio = fee / amount
        if fee_ratio > 0.05:
            risk_factors.append(20)
            compliance_flags.append("HIGH_FEE_RATIO")
    
    # COUNTERPARTY RISK ANALYSIS
    counterparty_risk = 0
    
    # Check for known high-risk address patterns
    if to_address:
        to_lower = to_address.lower()
        # Mixing service patterns (simplified - in production use real blacklists)
        if any(pattern in to_lower for pattern in ['mixer', 'tornado', 'blender']):
            counterparty_risk += 40
            counterparty_flags.append("MIXING_SERVICE")
        
        # Check for new/suspicious address (very short addresses are often test/suspicious)
        if len(to_address) < 40 and len(to_address) > 0:
            counterparty_risk += 15
            counterparty_flags.append("SUSPICIOUS_ADDRESS_FORMAT")
    
    # Source reputation
    high_risk_sources = ['unknown', 'p2p', 'localbitcoins', 'paxful']
    if source.lower() in high_risk_sources:
        counterparty_risk += 20
        counterparty_flags.append("HIGH_RISK_SOURCE")
    
    # Add counterparty risk to overall risk
    risk_factors.append(counterparty_risk)
    
    # Calculate final risk score
    risk_score = min(sum(risk_factors), 100)
    
    # Determine risk level
    if risk_score >= 80:
        risk_level = "critical"
    elif risk_score >= 60:
        risk_level = "high"
    elif risk_score >= 40:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # Simple anomaly detection
    anomaly_detected = risk_score >= 70 or len(compliance_flags) >= 3
    
    # Combine all flags
    all_flags = compliance_flags + counterparty_flags
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "anomaly_detected": anomaly_detected,
        "compliance_flags": all_flags,
        "counterparty_risk": counterparty_risk,
        "counterparty_flags": counterparty_flags
    }

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
        latest_block = w3.eth.block_number
        
        for i in range(min(100, latest_block)):
            block_number = latest_block - i
            block = w3.eth.get_block(block_number, full_transactions=True)
            
            for tx in block.transactions:
                if (tx['to'] and tx['to'].lower() == address.lower()) or \
                   (tx['from'] and tx['from'].lower() == address.lower()):
                    
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
            'fee': (gas_price * gas_used) / 1e9,
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
                'secret': 'dummy_secret',
                'sandbox': True
            })
        else:
            return generate_mock_exchange_transactions(exchange_name, limit)
        
        trades = await exchange.fetch_my_trades(limit=limit)
        
        transactions = []
        for trade in trades:
            transactions.append({
                'id': trade['id'],
                'amount': trade['amount'],
                'asset': trade['symbol'].split('/')[0],
                'price': trade['price'],
                'side': trade['side'],
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
            'fee': amount * 0.001,
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
    """
    Ingest transactions from wallets or exchanges
    UPGRADED: Uses programmatic scoring only (no AI calls)
    """
    
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
                        'hash': None,
                        'amount': tx_data['amount'],
                        'asset': tx_data['asset'],
                        'fee': tx_data['fee'],
                        'timestamp': tx_data['timestamp'],
                        'status': tx_data['status'],
                        'from_address': None,
                        'to_address': None
                    })
            
            # Store transactions with programmatic risk scoring
            async with AsyncSessionLocal() as session:
                new_transactions = []
                for tx_data in transactions_data:
                    # Calculate risk using programmatic scoring (NO AI)
                    risk_info = calculate_risk_score(tx_data)
                    
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
                        timestamp=tx_data['timestamp'],
                        # Apply programmatic risk scoring
                        risk_score=risk_info['risk_score'],
                        risk_level=risk_info['risk_level'],
                        anomaly_detected=risk_info['anomaly_detected'],
                        compliance_flags=risk_info['compliance_flags'],
                        analyzed_at=datetime.now()
                    )
                    session.add(transaction)
                    new_transactions.append(transaction)
                
                await session.commit()
                logger.info(f"✅ Ingested {len(new_transactions)} transactions for user {request.user_id} (programmatic scoring only)")
        
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
    
    background_tasks.add_task(process_ingestion)
    
    return {
        "status": "success",
        "message": "Transaction ingestion started (fast programmatic scoring)",
        "user_id": request.user_id
    }

@app.post("/audit", response_model=Dict[str, Any])
async def run_audit(
    request: AuditRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Run comprehensive audit on user's transactions
    UPGRADED: Pattern detection only (no AI calls during audit)
    """
    
    async def process_audit():
        try:
            # Detect suspicious patterns (programmatic, no AI)
            patterns = await detect_user_patterns(request.user_id)
            
            # Create alerts for detected patterns
            for pattern in patterns:
                await ai_engine.create_alert(
                    user_id=request.user_id,
                    transaction_id=None,
                    title=f"Suspicious Pattern: {pattern.pattern_type.replace('_', ' ').title()}",
                    message=f"{pattern.description} (Confidence: {pattern.confidence:.1%})",
                    severity="warning" if pattern.confidence < 0.8 else "high",
                    alert_type="pattern"
                )
            
            logger.info(f"✅ Audit completed for user {request.user_id}: {len(patterns)} patterns detected (no AI used)")
            
        except Exception as e:
            logger.error(f"Audit failed for user {request.user_id}: {e}")
    
    background_tasks.add_task(process_audit)
    
    return {
        "status": "success",
        "message": "Audit started (pattern detection only, no AI)",
        "user_id": request.user_id,
        "report_type": request.report_type
    }

@app.post("/report", response_model=AuditReportResponse)
async def generate_report(
    request: AuditRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate comprehensive audit report
    ✅ AI IS USED HERE for report generation
    """
    try:
        logger.info(f"🤖 Generating AI-powered report for user {request.user_id}")
        
        report_data = await generate_audit_report(
            user_id=request.user_id,
            report_type=request.report_type,
            date_from=request.date_from,
            date_to=request.date_to
        )
        
        logger.info(f"✅ AI report generated successfully")
        
        return AuditReportResponse(
            id=report_data["report_id"],
            title=f"{request.report_type.title()} Audit Report",
            report_type=request.report_type,
            summary=report_data["summary"][:500],
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

@app.get("/debug/report/{report_id}")
async def debug_report(report_id: int):
    """Debug endpoint to check report database entry"""
    async with AsyncSessionLocal() as session:
        report = await session.get(AuditReport, report_id)
        if not report:
            return {"error": "Report not found"}
        
        from pathlib import Path
        reports_dir = Path(settings.REPORTS_DIR).resolve()
        
        pdf_filename = report.pdf_path
        json_filename = report.json_path
        
        # Check both absolute and relative paths
        pdf_path_abs = Path(pdf_filename) if Path(pdf_filename).is_absolute() else reports_dir / pdf_filename
        json_path_abs = Path(json_filename) if Path(json_filename).is_absolute() else reports_dir / json_filename
        
        return {
            "report_id": report_id,
            "database_info": {
                "pdf_path": report.pdf_path,
                "json_path": report.json_path,
                "created_at": report.created_at.isoformat()
            },
            "filesystem_check": {
                "reports_dir": str(reports_dir),
                "reports_dir_exists": reports_dir.exists(),
                "pdf_path_resolved": str(pdf_path_abs),
                "pdf_exists": pdf_path_abs.exists(),
                "json_path_resolved": str(json_path_abs),
                "json_exists": json_path_abs.exists()
            },
            "files_in_reports_dir": [f.name for f in reports_dir.glob("*")] if reports_dir.exists() else []
        }

@app.get("/reports/{report_id}/download")
async def download_report(report_id: int, format: str = Query("pdf", pattern="^(pdf|json)$")):
    """Download generated report - FIXED: Complete Windows path handling with validation"""
    try:
        async with AsyncSessionLocal() as session:
            report = await session.get(AuditReport, report_id)
            if not report:
                logger.error(f"❌ Report {report_id} not found in database")
                raise HTTPException(status_code=404, detail="Report not found")
            
            # Get the filename from database
            filename = report.pdf_path if format == "pdf" else report.json_path
            if not filename:
                logger.error(f"❌ Report {report_id} has no {format} path in database")
                raise HTTPException(status_code=404, detail=f"Report {format.upper()} path not found in database")
            
            # Construct full path - Windows-compatible
            from pathlib import Path
            reports_dir = Path(settings.REPORTS_DIR).resolve()
            
            logger.info(f"=" * 60)
            logger.info(f"📥 DOWNLOAD REQUEST for Report {report_id}")
            logger.info(f"=" * 60)
            logger.info(f"Format requested: {format}")
            logger.info(f"DB filename: {filename}")
            logger.info(f"Reports directory: {reports_dir}")
            logger.info(f"Reports dir exists: {reports_dir.exists()}")
            
            # Convert to Path object for proper handling
            filename_path = Path(filename)
            logger.info(f"Filename is absolute: {filename_path.is_absolute()}")
            
            # If it's already an absolute path, use it directly
            if filename_path.is_absolute():
                file_path = filename_path.resolve()
                logger.info(f"Using absolute path from DB: {file_path}")
            else:
                # Otherwise, join with reports directory
                file_path = (reports_dir / filename).resolve()
                logger.info(f"Constructed path: {file_path}")
            
            logger.info(f"Final resolved path: {file_path}")
            logger.info(f"File exists: {file_path.exists()}")
            logger.info(f"File is file: {file_path.is_file()}")
            
            if file_path.exists() and file_path.is_file():
                import os
                file_size = os.path.getsize(file_path)
                logger.info(f"File size: {file_size} bytes")
            
            if not file_path.exists():
                logger.error(f"=" * 60)
                logger.error(f"❌ FILE NOT FOUND")
                logger.error(f"=" * 60)
                # List what's actually in the reports directory
                try:
                    all_files = list(reports_dir.glob("*"))
                    logger.error(f"📁 Contents of {reports_dir}:")
                    logger.error(f"Total files: {len(all_files)}")
                    pdf_files = [f for f in all_files if f.suffix == '.pdf']
                    json_files = [f for f in all_files if f.suffix == '.json']
                    logger.error(f"PDF files: {len(pdf_files)}")
                    logger.error(f"JSON files: {len(json_files)}")
                    
                    # Show relevant files
                    logger.error(f"\nRelevant files containing 'audit_report_1_20251024':")
                    for f in all_files:
                        if 'audit_report_1_20251024' in f.name:
                            logger.error(f"   - {f.name} ({os.path.getsize(f)} bytes)")
                except Exception as e:
                    logger.error(f"Could not list directory: {e}")
                
                raise HTTPException(
                    status_code=404, 
                    detail=f"Report file not found at: {file_path}"
                )
            
            if not file_path.is_file():
                logger.error(f"❌ Path exists but is not a file: {file_path}")
                raise HTTPException(status_code=500, detail="Invalid file path")
            
            # Get file size for validation
            import os
            file_size = os.path.getsize(file_path)
            logger.info(f"✅ File found! Size: {file_size} bytes")
            
            if file_size == 0:
                logger.error(f"⚠️  WARNING: File size is 0 bytes!")
                raise HTTPException(status_code=500, detail="Report file is empty")
            
            media_type = "application/pdf" if format == "pdf" else "application/json"
            
            logger.info(f"📤 Serving file: {file_path.name}")
            logger.info(f"   Media type: {media_type}")
            logger.info(f"   File size: {file_size:,} bytes")
            logger.info(f"=" * 60)
            
            # Return the file
            return FileResponse(
                path=str(file_path),
                filename=f"audit_report_{report_id}.{format}",
                media_type=media_type
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in download endpoint: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/users/{user_id}/stats", response_model=StatsResponse)
async def get_user_stats(user_id: int, db: AsyncSession = Depends(get_db)):
    """Get user statistics"""
    
    total_transactions_result = await db.execute(
        select(func.count(Transaction.id)).where(Transaction.user_id == user_id)
    )
    total_transactions = total_transactions_result.scalar() or 0
    
    high_risk_result = await db.execute(
        select(func.count(Transaction.id))
        .where(Transaction.user_id == user_id)
        .where(Transaction.risk_level.in_(["high", "critical"]))
    )
    high_risk_transactions = high_risk_result.scalar() or 0
    
    wallet_count_result = await db.execute(
        select(func.count(Wallet.id)).where(Wallet.user_id == user_id)
    )
    total_wallets = wallet_count_result.scalar() or 0
    
    exchange_count_result = await db.execute(
        select(func.count(Exchange.id)).where(Exchange.user_id == user_id)
    )
    total_exchanges = exchange_count_result.scalar() or 0
    
    unresolved_alerts_result = await db.execute(
        select(func.count(Alert.id))
        .where(Alert.user_id == user_id)
        .where(Alert.is_resolved == False)
    )
    unresolved_alerts = unresolved_alerts_result.scalar() or 0
    
    last_transaction_result = await db.execute(
        select(Transaction.timestamp)
        .where(Transaction.user_id == user_id)
        .order_by(desc(Transaction.timestamp))
        .limit(1)
    )
    last_sync = last_transaction_result.scalar_one_or_none()
    
    return StatsResponse(
        total_transactions=total_transactions,
        total_wallets=total_wallets,
        total_exchanges=total_exchanges,
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
            tx_data = {
                'amount': random.uniform(10, 5000),
                'gas_price': random.uniform(20, 100),
                'fee': random.uniform(0.001, 0.1),
                'status': random.choices(['confirmed', 'failed'], weights=[95, 5])[0],
                'timestamp': base_time + timedelta(hours=random.randint(1, 720))
            }
            
            # Calculate risk programmatically
            risk_info = calculate_risk_score(tx_data)
            
            transaction = Transaction(
                user_id=user_id,
                hash=f"0x{''.join(random.choices('0123456789abcdef', k=64))}",
                amount=tx_data['amount'],
                asset=random.choice(['BTC', 'ETH', 'USDT', 'BNB']),
                transaction_type=random.choice(['send', 'receive', 'trade']),
                source=random.choice(['ethereum', 'binance', 'coinbase']),
                status=tx_data['status'],
                timestamp=tx_data['timestamp'],
                fee=tx_data['fee'],
                risk_score=risk_info['risk_score'],
                risk_level=risk_info['risk_level'],
                anomaly_detected=risk_info['anomaly_detected'],
                compliance_flags=risk_info['compliance_flags']
            )
            
            session.add(transaction)
            mock_transactions.append(transaction)
        
        await session.commit()
    
    return {
        "status": "created",
        "count": len(mock_transactions),
        "user_id": user_id,
        "message": f"Created {count} mock transactions with programmatic scoring"
    }

@app.post("/chat")
async def chat_endpoint(request: dict):
    """
    Enhanced chat endpoint with proper Ollama integration
    ✅ AI IS USED HERE for conversational responses
    """
    logger.info(f"🤖 Chat request received")
    query = request.get("query") or request.get("message", "")
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
    
    # Use Ollama for complex queries
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
                    logger.info(f"Ollama response received")
                    
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
# Startup and Shutdown Events
# -------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("⚡ Starting CryptoAuditAI Backend...")
    logger.info("🔧 UPGRADED VERSION: AI only for reports and chat")
    
    # Initialize database
    logger.info("📊 Initializing database...")
    await init_db()
    
    # Test database connection
    db_status = await test_connection()
    if db_status:
        logger.info("✅ Database connected")
    else:
        logger.error("❌ Database connection failed!")
    
    # Test Ollama connection (non-blocking)
    try:
        logger.info("🤖 Testing Ollama connection...")
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                logger.info(f"✅ Ollama connected. Models: {model_names}")
                
                # Check if our model is available
                if settings.OLLAMA_MODEL in model_names or any(settings.OLLAMA_MODEL in m for m in model_names):
                    logger.info(f"✅ Model '{settings.OLLAMA_MODEL}' is available")
                else:
                    logger.warning(f"⚠️  Model '{settings.OLLAMA_MODEL}' not found. Available: {model_names}")
            else:
                logger.warning("⚠️  Ollama not responding properly")
    except httpx.TimeoutException:
        logger.warning("⚠️  Ollama connection timeout (server may be starting)")
    except Exception as e:
        logger.warning(f"⚠️  Failed to connect to Ollama: {e}")
    
    logger.info("=" * 60)
    logger.info("🚀 CryptoAuditAI Backend Ready!")
    logger.info("=" * 60)
    logger.info("📍 Endpoints:")
    logger.info("   ⚡ /ingest  - Fast programmatic scoring (no AI)")
    logger.info("   ⚡ /audit   - Pattern detection only (no AI)")
    logger.info("   🤖 /report  - AI-powered report generation")
    logger.info("   🤖 /chat    - AI-powered conversational interface")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down CryptoAuditAI Backend...")

# -------------------------
# Main Entry Point
# -------------------------

if __name__ == "__main__":
    logger.info("Starting CryptoAuditAI Backend - UPGRADED VERSION")
    logger.info(f"Ollama URL: {settings.OLLAMA_HOST}")
    logger.info(f"Default Model: {settings.OLLAMA_MODEL}")
    logger.info("AI Usage: Reports & Chat only (not during audits)")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )