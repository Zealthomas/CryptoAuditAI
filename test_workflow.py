# test_workflow.py
import asyncio
import logging
from sqlalchemy import select
from database import init_db, AsyncSessionLocal
from models import Document, User
from ingest import ingest_urls
from retrieval import RetrievalEngine
from ai_engine import agentic_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # 1️⃣ Init DB
    logger.info("Initializing database...")
    await init_db()

    # 2️⃣ Insert a test user (if not exists)
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(User).where(User.id == 1))
        user = res.scalar_one_or_none()
        if not user:
            user = User(username="tester", email="tester@example.com", password_hash="hashed_pw")
            session.add(user)
            await session.commit()
            logger.info("✅ Test user created")
        else:
            logger.info("ℹ️ Test user already exists")

    # 3️⃣ Ingest URLs
    urls = ["https://docs.python.org/3/", "https://fastapi.tiangolo.com/"]
    logger.info(f"Ingesting {urls}...")
    stats = await ingest_urls(urls)
    logger.info(f"✅ Ingestion stats: {stats}")

    # 4️⃣ Query CRUD layer for documents
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Document))
        docs = result.scalars().all()
        logger.info(f"✅ Retrieved {len(docs)} documents from DB")
        for doc in docs[:2]:  # show first 2
            logger.info(f"Doc: id={doc.id}, url={doc.url}, size={len(doc.content)} chars")

    # 5️⃣ Index + Query Retrieval
    engine = RetrievalEngine()
    logger.info("Indexing transactions for user=1...")
    count = await engine.index_transactions(user_id=1)
    logger.info(f"✅ Indexed {count} transactions")

    results = await engine.query(1, "high risk transactions", top_k=3)
    logger.info("✅ Retrieval Results:")
    for score, tx in results:
        logger.info(f"- Score={score:.3f}, TxHash={tx.hash}, Risk={tx.risk_level}")

    # 6️⃣ Chat test
    query = "Summarize risks in my wallet"
    logger.info(f"Asking agent: {query}")
    answer = await agentic_answer(user_id=1, query=query)
    logger.info(f"✅ Chat response: {answer}")

if __name__ == "__main__":
    asyncio.run(main())
