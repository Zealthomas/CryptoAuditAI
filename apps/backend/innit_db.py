# apps/backend/init_db.py

import asyncio
from datetime import datetime
from database import engine
from models import Base, User, Wallet, Exchange

async def init_db():
    async with engine.begin() as conn:
        print("Dropping all tables (if any exist)...")
        await conn.run_sync(Base.metadata.drop_all)
        print("Creating all tables...")
        await conn.run_sync(Base.metadata.create_all)

        # Insert initial test data
        print("Creating test user, wallet, and exchange...")
        await conn.run_sync(_create_test_data)

    print("✅ Database initialized with test user, wallet, and exchange!")

def _create_test_data(sync_conn):
    # sync_conn is a standard SQLAlchemy connection inside run_sync
    # Use normal session style with connection
    from sqlalchemy.orm import Session
    session = Session(sync_conn)

    # Test user
    user = User(email="test@example.com", hashed_password="hashedpassword123")
    session.add(user)
    session.flush()  # populates user.id

    # Test wallet
    wallet = Wallet(
        user_id=user.id,
        address="0x1234567890abcdef1234567890abcdef12345678",
        label="Test Wallet",
        last_synced=datetime.utcnow()
    )
    session.add(wallet)

    # Test exchange
    exchange = Exchange(
        user_id=user.id,
        name="binance",
        api_key_encrypted="testkey",
        api_secret_encrypted="testsecret",
        is_sandbox=True,
        last_synced=datetime.utcnow()
    )
    session.add(exchange)

    session.commit()
    session.close()

if __name__ == "__main__":
    asyncio.run(init_db())
