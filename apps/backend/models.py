"""
Complete CryptoAuditAI Models
Updated for crypto transaction auditing with proper relationships
"""
from datetime import datetime
from typing import Optional, List, Dict
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, Boolean, JSON
from sqlalchemy.orm import relationship
from database import Base
from pydantic import BaseModel, Field
import bcrypt

# -------------------------
# SQLAlchemy ORM MODELS
# -------------------------

class User(Base):
    __tablename__ = "users"
   
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
   
    # Relationships
    wallets = relationship("Wallet", back_populates="user", cascade="all, delete-orphan")
    exchanges = relationship("Exchange", back_populates="user", cascade="all, delete-orphan")
    audit_reports = relationship("AuditReport", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
   
    def set_password(self, password: str):
        """Hashes a password and stores it securely."""
        self.password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
   
    def verify_password(self, password: str) -> bool:
        """Verifies a password against the stored hash."""
        return bcrypt.checkpw(password.encode("utf-8"), self.password_hash.encode("utf-8"))


class Wallet(Base):
    __tablename__ = "wallets"
   
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    address = Column(String(128), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=True)  # User-friendly name
    balance = Column(Float, default=0.0)
    blockchain = Column(String(20), default="ethereum")  # ethereum, bitcoin, etc
    created_at = Column(DateTime, default=datetime.utcnow)
    last_synced = Column(DateTime, nullable=True)
   
    # Relationships
    user = relationship("User", back_populates="wallets")
    transactions = relationship("Transaction", back_populates="wallet", cascade="all, delete-orphan")


class Exchange(Base):
    __tablename__ = "exchanges"
   
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)  # binance, coinbase, etc
    api_key_encrypted = Column(String(512), nullable=True)  # Encrypted API key
    api_secret_encrypted = Column(String(512), nullable=True)  # Encrypted secret
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_synced = Column(DateTime, nullable=True)
   
    # Relationships
    user = relationship("User", back_populates="exchanges")
    transactions = relationship("Transaction", back_populates="exchange", cascade="all, delete-orphan")


class Transaction(Base):
    __tablename__ = "transactions"
   
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # Direct user reference
    wallet_id = Column(Integer, ForeignKey("wallets.id"), nullable=True)  # For on-chain
    exchange_id = Column(Integer, ForeignKey("exchanges.id"), nullable=True)  # For exchange
   
    # Transaction Details
    hash = Column(String(128), unique=True, nullable=True, index=True)  # On-chain hash
    amount = Column(Float, nullable=False)
    asset = Column(String(20), nullable=False)  # BTC, ETH, USDT, etc
    from_address = Column(String(128), nullable=True)
    to_address = Column(String(128), nullable=True)
   
    # Transaction Type & Status
    transaction_type = Column(String(20), nullable=False)  # send, receive, trade, swap
    status = Column(String(20), default="confirmed")  # pending, confirmed, failed
    source = Column(String(50), nullable=False)  # ethereum, binance, coinbase, etc
   
    # Fees & Gas
    fee = Column(Float, default=0.0)
    gas_price = Column(Float, nullable=True)
    gas_used = Column(Integer, nullable=True)
   
    # Risk Assessment Fields
    risk_level = Column(String(20), default="low")  # low, medium, high, critical
    risk_score = Column(Float, default=0.0)  # 0-100 risk score
    anomaly_detected = Column(Boolean, default=False)
    compliance_flags = Column(JSON, nullable=True)  # Store compliance issues as JSON
   
    # Timestamps
    timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    analyzed_at = Column(DateTime, nullable=True)  # When AI analysis was done
   
    # Relationships
    wallet = relationship("Wallet", back_populates="transactions")
    exchange = relationship("Exchange", back_populates="transactions")


class AuditReport(Base):
    __tablename__ = "audit_reports"
   
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(200), nullable=False)
    report_type = Column(String(50), default="comprehensive")  # comprehensive, risk, compliance
   
    # Report Content
    summary = Column(Text, nullable=True)  # Executive summary
    full_report = Column(Text, nullable=False)  # Full AI-generated report
    recommendations = Column(JSON, nullable=True)  # Structured recommendations
   
    # Risk Metrics
    overall_risk_score = Column(Float, default=0.0)
    total_transactions = Column(Integer, default=0)
    flagged_transactions = Column(Integer, default=0)
    compliance_score = Column(Float, default=0.0)
    total_risk_score = Column(Float, default=0.0)
    average_risk_score = Column(Float, default=0.0)

   
    # File paths
    pdf_path = Column(String(500), nullable=True)
    json_path = Column(String(500), nullable=True)
   
    created_at = Column(DateTime, default=datetime.utcnow)
   
    # Relationships
    user = relationship("User", back_populates="audit_reports")


class Alert(Base):
    __tablename__ = "alerts"
   
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=True)
   
    # Alert Details
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(String(20), default="info")  # info, warning, high, critical
    alert_type = Column(String(50), nullable=False)  # anomaly, compliance, risk, suspicious
   
    # Status
    is_read = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
   
    created_at = Column(DateTime, default=datetime.utcnow)
   
    # Relationships
    user = relationship("User", back_populates="alerts")


# -------------------------
# Pydantic Request/Response Models
# -------------------------

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
   
    class Config:
        from_attributes = True


class WalletCreate(BaseModel):
    address: str = Field(..., min_length=26, max_length=128)
    name: Optional[str] = None
    blockchain: str = "ethereum"


class WalletResponse(BaseModel):
    id: int
    address: str
    name: Optional[str]
    balance: float
    blockchain: str
    created_at: datetime
    last_synced: Optional[datetime]
   
    class Config:
        from_attributes = True


class ExchangeCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    api_key: Optional[str] = None
    api_secret: Optional[str] = None


class ExchangeResponse(BaseModel):
    id: int
    name: str
    is_active: bool
    created_at: datetime
    last_synced: Optional[datetime]
   
    class Config:
        from_attributes = True


class TransactionResponse(BaseModel):
    id: int
    hash: Optional[str]
    amount: float
    asset: str
    from_address: Optional[str]
    to_address: Optional[str]
    transaction_type: str
    status: str
    source: str
    risk_level: str
    risk_score: float
    anomaly_detected: bool
    timestamp: datetime
   
    class Config:
        from_attributes = True


class IngestRequest(BaseModel):
    user_id: int
    wallet_address: Optional[str] = None
    exchange_name: Optional[str] = None
    limit: int = Field(default=100, le=1000)


class AuditRequest(BaseModel):
    user_id: int
    report_type: str = "comprehensive"
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


class AuditReportResponse(BaseModel):
    id: int
    title: str
    report_type: str
    summary: Optional[str]
    overall_risk_score: float
    total_transactions: int
    flagged_transactions: int
    compliance_score: float
    created_at: datetime
    pdf_path: Optional[str]
    json_path: Optional[str]
   
    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    id: int
    title: str
    message: str
    severity: str
    alert_type: str
    is_read: bool
    is_resolved: bool
    created_at: datetime
   
    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    total_transactions: int
    total_wallets: int
    total_exchanges: int
    high_risk_transactions: int
    unresolved_alerts: int
    last_sync: Optional[datetime]

    # New fields (frontend-friendly)
    transactions_by_risk: Dict[str, int] = Field(
        default_factory=lambda: {"low": 0, "medium": 0, "high": 0, "critical": 0}
    )
    generated_at: Optional[datetime] = None

    class Config:
        from_attributes = True