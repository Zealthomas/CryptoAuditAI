# reset_db.py
import os
import asyncio
import logging

from database import engine, Base
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_FILE = "test.db"


async def reset_database():
    # 1. Delete old DB if it exists
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        logger.info(f"🗑️ Deleted old database file: {DB_FILE}")

    # 2. Recreate tables from models.py
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database reset complete. Fresh schema applied.")


if __name__ == "__main__":
    asyncio.run(reset_database())
