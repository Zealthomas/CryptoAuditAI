# apps/backend/database.py
"""
Database setup and encryption utilities for CryptoAuditAI

Features:
- Async SQLAlchemy engine and session
- Dependency-ready session for FastAPI
- Encryption/decryption helpers for API keys
- Init and health checks with logging
- Future-proof: performance logging and dynamic table creation support
"""

import logging
import base64
import time
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text
from cryptography.fernet import Fernet, InvalidToken
from config import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------
# SQLAlchemy Base & Engine
# ---------------------------
Base = declarative_base()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
    class_=AsyncSession,
)

# ---------------------------
# Dependency for FastAPI
# ---------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async generator dependency for FastAPI routes.
    Yields a session and ensures proper rollback and closure.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session rollback due to: {e}")
            raise
        finally:
            await session.close()

# ---------------------------
# Encryption helpers
# ---------------------------
def get_cipher() -> Fernet:
    """
    Build a Fernet cipher from ENCRYPTION_KEY (32-byte string).
    """
    key = settings.ENCRYPTION_KEY.get_secret_value()
    if len(key) != 32:
        raise ValueError("ENCRYPTION_KEY must be 32 bytes long")
    key_b64 = base64.urlsafe_b64encode(key.encode())
    return Fernet(key_b64)

def encrypt_api_key(api_key: str) -> str:
    cipher = get_cipher()
    return cipher.encrypt(api_key.encode()).decode()

def decrypt_api_key(encrypted_key: str) -> str:
    cipher = get_cipher()
    try:
        return cipher.decrypt(encrypted_key.encode()).decode()
    except InvalidToken:
        raise ValueError("Invalid encryption key or corrupted data")

# ---------------------------
# Init & Health
# ---------------------------
async def init_db():
    """
    Create all tables defined in models.Base.
    Note: This imports models to ensure they're registered with Base.metadata
    """
    # Import models here to avoid circular imports
    from models import User, Wallet, Exchange, Transaction, AuditReport, Alert
    
    start = time.time()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info(f"✅ Database initialized (all tables created) in {time.time() - start:.2f}s")

async def test_connection() -> bool:
    """
    Verify database connectivity by running a simple SELECT 1.
    """
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

# ---------------------------
# Database reset utility
# ---------------------------
async def reset_database():
    """
    Drop all tables and recreate them with proper schema.
    WARNING: This will delete all data!
    """
    logger.warning("🚨 Resetting database - all data will be lost!")
    
    async with engine.begin() as conn:
        # Drop all tables
        await conn.run_sync(Base.metadata.drop_all)
        logger.info("🗑️ All tables dropped")
    
    # Recreate tables
    await init_db()
    logger.info("✅ Database reset complete")

# ---------------------------
# Optional: helper to dynamically create a table at runtime
# ---------------------------
async def create_table_dynamically(table_class):
    """
    Dynamically create a table from a SQLAlchemy declarative model.
    """
    async with engine.begin() as conn:
        await conn.run_sync(table_class.metadata.create_all)
    logger.info(f"✅ Table {table_class.__name__} created dynamically")