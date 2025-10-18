# apps/backend/test_retrieval.py
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from retrieval import RetrievalEngine
from database import AsyncSessionLocal, init_db
from models import Transaction, User, Wallet, Exchange

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_test_data():
    """Create test data that matches your exact schema requirements"""
    async with AsyncSessionLocal() as session:
        # Clear any existing test data
        await session.execute(Transaction.__table__.delete().where(Transaction.user_id.in_([1, 2])))
        await session.execute(User.__table__.delete().where(User.id.in_([1, 2])))
        await session.execute(Wallet.__table__.delete().where(Wallet.user_id.in_([1, 2])))
        await session.execute(Exchange.__table__.delete().where(Exchange.user_id.in_([1, 2])))
        await session.commit()

        # Create test users
        test_users = [
            User(
                id=1,
                email="test1@example.com",
                hashed_password="hashed_password_1"
            ),
            User(
                id=2, 
                email="test2@example.com",
                hashed_password="hashed_password_2"
            )
        ]

        # Create test wallets
        test_wallets = [
            Wallet(
                user_id=1,
                address="0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                label="Test Wallet 1"
            ),
            Wallet(
                user_id=2,
                address="0x742d35Cc6634C0532925a3b844Bc454e4438f44f", 
                label="Test Wallet 2"
            )
        ]

        # Create test exchanges
        test_exchanges = [
            Exchange(
                user_id=1,
                name="binance",
                api_key_encrypted="encrypted_api_key_1",
                api_secret_encrypted="encrypted_api_secret_1"
            ),
            Exchange(
                user_id=2,
                name="bybit",
                api_key_encrypted="encrypted_api_key_2",
                api_secret_encrypted="encrypted_api_secret_2"
            )
        ]

        # Create test transactions with ALL required fields
        test_transactions = [
            Transaction(
                hash="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                user_id=1,
                wallet_id=1,
                exchange_id=1,
                from_address="0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                to_address="0x742d35Cc6634C0532925a3b844Bc454e4438f44f",
                value="1.5",
                gas_used="21000",
                gas_price="0.00000005",
                timestamp=datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                source="binance",
                risk_score=0.8,
                risk_level="high",
                anomaly_detected=True
            ),
            Transaction(
                hash="0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
                user_id=1,
                wallet_id=1,
                exchange_id=1,
                from_address="0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                to_address="0x742d35Cc6634C0532925a3b844Bc454e4438f44f",
                value="0.5",
                gas_used="21000",
                gas_price="0.00000004",
                timestamp=datetime(2023, 1, 2, 11, 0, 0, tzinfo=timezone.utc),
                source="binance",
                risk_score=0.2,
                risk_level="low",
                anomaly_detected=False
            ),
            Transaction(
                hash="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                user_id=2,
                wallet_id=2,
                exchange_id=2,
                from_address="0x742d35Cc6634C0532925a3b844Bc454e4438f44f",
                to_address="0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                value="10.0",
                gas_used="42000",
                gas_price="0.00000006",
                timestamp=datetime(2023, 1, 3, 12, 0, 0, tzinfo=timezone.utc),
                source="bybit",
                risk_score=0.9,
                risk_level="critical",
                anomaly_detected=True
            )
        ]

        session.add_all(test_users)
        session.add_all(test_wallets)
        session.add_all(test_exchanges)
        session.add_all(test_transactions)
        
        await session.commit()
        logger.info("Created complete test data with users, wallets, exchanges, and transactions")

async def test_indexing():
    """Test indexing functionality"""
    logger.info("Testing indexing...")
    
    engine = RetrievalEngine()
    
    # Test indexing for user 1
    count = await engine.index_transactions(1)
    logger.info(f"Indexed {count} transactions for user 1")
    
    # Test indexing for user 2
    count = await engine.index_transactions(2)
    logger.info(f"Indexed {count} transactions for user 2")
    
    # Test indexing again (should find no new transactions)
    count = await engine.index_transactions(1)
    logger.info(f"Second indexing found {count} new transactions")

async def test_query():
    """Test query functionality"""
    logger.info("Testing query...")
    
    engine = RetrievalEngine()
    
    # Test query for user 1
    results = await engine.query(1, "high risk binance transaction", top_k=3)
    logger.info(f"Query found {len(results)} results for user 1")
    
    for score, transaction in results:
        logger.info(f"Score: {score:.4f} - Hash: {transaction.hash} - Risk: {transaction.risk_level}")
    
    # Test query for user 2
    results = await engine.query(2, "bybit large transfer", top_k=3)
    logger.info(f"Query found {len(results)} results for user 2")
    
    # Test query for non-existent user
    results = await engine.query(999, "any query", top_k=3)
    logger.info(f"Query results for non-existent user: {len(results)} transactions")

async def main():
    """Run all tests"""
    try:
        # Initialize database
        await init_db()
        
        # Setup complete test data
        await setup_test_data()
        
        # Run tests
        await test_indexing()
        await test_query()
        
        logger.info("✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())