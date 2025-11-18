# test_db.py
import asyncio
from database import init_db, test_connection, encrypt_api_key, decrypt_api_key

async def main():
    print("🔹 Initializing DB...")
    await init_db()

    print("🔹 Testing DB connection...")
    success = await test_connection()
    print(f"Connection successful: {success}")

    print("🔹 Testing encryption...")
    key = "my-secret-api-key"
    encrypted = encrypt_api_key(key)
    print(f"Encrypted: {encrypted}")
    decrypted = decrypt_api_key(encrypted)
    print(f"Decrypted: {decrypted}")

    assert decrypted == key, "Encryption/decryption failed!"

if __name__ == "__main__":
    asyncio.run(main())
