import asyncio
from database import engine
from sqlalchemy import text

async def check_tables():
    async with engine.begin() as conn:
        result = await conn.execute(text('SELECT name FROM sqlite_master WHERE type=\'table\';'))
        tables = [row[0] for row in result]
        print('Available tables:', tables)
        
asyncio.run(check_tables())