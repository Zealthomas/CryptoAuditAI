# apps/backend/mock_data.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, User, Wallet, Exchange, Transaction, AuditReport, Alert
from datetime import datetime, timedelta
import random
import secrets
import string
from enum import Enum

# Configuration
DB_URL = "sqlite:///test.db"
MOCK_RANGE = (0, 4)  # Generate 0-4 records per table

# Enums matching your schema
class MockRiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MockSource(Enum):
    ETHEREUM = "ethereum"
    BINANCE = "binance"
    BYBIT = "bybit"

class MockAlertType(Enum):
    ANOMALY = "anomaly"
    COMPLIANCE = "compliance"
    RISK = "risk"

# Helper functions
def generate_eth_address():
    return "0x" + secrets.token_hex(20)

def generate_tx_hash():
    return "0x" + secrets.token_hex(32)

def random_date(days_back=30):
    return datetime.now() - timedelta(days=random.randint(1, days_back))

def generate_mock_data():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Clear existing mocks
        session.query(Alert).delete()
        session.query(AuditReport).delete()
        session.query(Transaction).delete()
        session.query(Wallet).delete()
        session.query(Exchange).delete()
        session.query(User).delete()
        session.commit()

        # Generate Users (1-4)
        users = []
        for i in range(1, random.randint(2, 5)):
            user = User(
                email=f"user{i}@mock.com",
                hashed_password=f"mock_hash_{secrets.token_hex(8)}"
            )
            session.add(user)
            users.append(user)
        session.commit()

        # Generate Wallets (0-4 per user)
        wallets = []
        for user in users:
            for w in range(random.randint(0, 4)):
                wallet = Wallet(
                    user_id=user.id,
                    address=generate_eth_address(),
                    label=f"Wallet {w+1}"
                )
                session.add(wallet)
                wallets.append(wallet)
        session.commit()

        # Generate Exchanges (0-2 per user)
        exchanges = []
        for user in users:
            for e in range(random.randint(0, 2)):
                exchange = Exchange(
                    user_id=user.id,
                    name=random.choice(list(MockSource)).value,
                    api_key_encrypted=f"mock_key_{secrets.token_hex(16)}",
                    api_secret_encrypted=f"mock_secret_{secrets.token_hex(32)}",
                    is_sandbox=random.choice([True, False])
                )
                session.add(exchange)
                exchanges.append(exchange)
        session.commit()

        # Generate Transactions (0-4 per wallet/exchange)
        for wallet in wallets:
            for t in range(random.randint(0, 4)):
                tx = Transaction(
                    hash=generate_tx_hash(),
                    user_id=wallet.user_id,
                    wallet_id=wallet.id,
                    from_address=wallet.address,
                    to_address=generate_eth_address(),
                    value=str(round(random.uniform(0.001, 10), 8)),
                    gas_used=str(random.randint(21000, 100000)),
                    gas_price=str(round(random.uniform(10, 100), 2)),
                    timestamp=random_date(),
                    source=MockSource.ETHEREUM.value,
                    risk_score=round(random.uniform(0, 1), 2),
                    risk_level=random.choice(list(MockRiskLevel)).value,
                    anomaly_detected=random.choice([True, False])
                )
                session.add(tx)

        for exchange in exchanges:
            for t in range(random.randint(0, 4)):
                tx = Transaction(
                    hash=generate_tx_hash(),
                    user_id=exchange.user_id,
                    exchange_id=exchange.id,
                    from_address=f"exchange_{exchange.name}",
                    to_address=generate_eth_address(),
                    value=str(round(random.uniform(0.01, 50), 2)),
                    timestamp=random_date(),
                    source=exchange.name,
                    risk_score=round(random.uniform(0, 1), 2),
                    risk_level=random.choice(list(MockRiskLevel)).value
                )
                session.add(tx)
        session.commit()

        # Generate Audit Reports (0-2 per user)
        for user in users:
            for a in range(random.randint(0, 2)):
                report = AuditReport(
                    audit_id=f"audit_{secrets.token_hex(8)}",
                    user_id=user.id,
                    status=random.choice(["processing", "completed", "error"]),
                    transaction_count=random.randint(10, 1000),
                    results={"mock": "data"},
                    summary={"mock": "summary"}
                )
                session.add(report)
        session.commit()

        # Generate Alerts (0-3 per user)
        for user in users:
            for a in range(random.randint(0, 3)):
                alert = Alert(
                    user_id=user.id,
                    transaction_hash=generate_tx_hash(),
                    alert_type=random.choice(list(MockAlertType)).value,
                    risk_level=random.choice(list(MockRiskLevel)).value,
                    message="Mock alert: " + " ".join(
                        random.choices(string.ascii_letters + string.digits, k=20)
                    )
                )
                session.add(alert)
        session.commit()

        print(f"✅ Generated mock data: {len(users)} users, {len(wallets)} wallets, "
              f"{len(exchanges)} exchanges, {session.query(Transaction).count()} transactions")

    except Exception as e:
        session.rollback()
        print(f"❌ Error generating mocks: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    generate_mock_data()