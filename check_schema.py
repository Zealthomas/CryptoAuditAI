import asyncio
from database import engine
from sqlalchemy import text

async def check_schema():
    async with engine.begin() as conn:
        # Check transactions table structure
        result = await conn.execute(text("PRAGMA table_info(transactions);"))
        columns = result.fetchall()
        
        print("Current transactions table columns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULL'}")
        
        print("\nSample data:")
        try:
            result = await conn.execute(text("SELECT * FROM transactions LIMIT 3;"))
            rows = result.fetchall()
            for row in rows:
                print(f"  {row}")
        except Exception as e:
            print(f"  No data or error: {e}")

if __name__ == "__main__":
    asyncio.run(check_schema())