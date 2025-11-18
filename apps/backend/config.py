import os
from typing import List
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

class Settings(BaseSettings):
    # Core Settings
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Database - Changed to regular string for direct access
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./test.db",
        env="DATABASE_URL"
    )
    
    # Security - Keep sensitive fields as SecretStr
    SECRET_KEY: SecretStr = Field(
        default=os.getenv("SECRET_KEY", ""),
        env="SECRET_KEY",
        min_length=32
    )
    ENCRYPTION_KEY: SecretStr = Field(
        default=os.getenv("ENCRYPTION_KEY", ""),
        env="ENCRYPTION_KEY",
        min_length=32,
        max_length=32
    )
    
    # Ollama Configuration
    OLLAMA_HOST: str = Field(
        default="http://localhost:11434",
        env="OLLAMA_HOST"
    )
    OLLAMA_MODEL: str = Field(
        default="mistral:instruct",
        env="OLLAMA_MODEL"
    )
    
    # Reports
    REPORTS_DIR: str = Field(
        default="./reports",
        env="REPORTS_DIR"
    )

    # Web3/Infura Configuration
    INFURA_PROJECT_ID: str = Field(
        default="",
        env="INFURA_PROJECT_ID",
        description="Infura project ID for Ethereum node access"
    )
    
    INFURA_PROJECT_SECRET: SecretStr = Field(
        default=SecretStr(""),
        env="INFURA_PROJECT_SECRET",
        description="Infura project secret"
    )
    
    ETHEREUM_NETWORK: str = Field(
        default="mainnet",
        env="ETHEREUM_NETWORK",
        description="Ethereum network (mainnet, ropsten, etc)"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# Validate required secrets
if not settings.SECRET_KEY.get_secret_value():
    raise ValueError("SECRET_KEY must be set in .env (min 32 chars)")
if not settings.ENCRYPTION_KEY.get_secret_value():
    raise ValueError("ENCRYPTION_KEY must be set in .env (exactly 32 chars)")

# Create reports directory
os.makedirs(settings.REPORTS_DIR, exist_ok=True)