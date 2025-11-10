from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from dotenv import load_dotenv
import os
import urllib.parse

load_dotenv()

class Settings(BaseSettings):
    """
    App settings loaded from environment variables or .env file
    """

    # API Keys
    LANCEDB_API_KEY: str | None = None
    COHERE_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    LANGCHAIN_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    LLM_API_KEY: str | None = None

    # LangSmith / LangChain
    LANGSMITH_TRACING: bool = True
    LANGCHAIN_PROJECT: str = "learnaly-tica"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # Vector DB
    LANCE_VECTOR_DB: str = "learnalytica"
    DATABASE_URL: Optional[str] = "sqlite:///./cognee_memory.db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Agent Endpoint
    AGENT_LEARNING_INSIGHT_ENDPOINT: str = "http://localhost:4040"
    AGENT_DATA_ANALYST_ENDPOINT: str = "http://localhost:4041"
    AGENT_REFLEXION_ENDPOINT: str = "http://localhost:4042"

    # ==================== Cognee Settings ====================
    COGNEE_MEMORY_MAX_SIZE: int = 100  # Max reflections per user
    COGNEE_CACHE_EXPIRY_HOURS: int = 24  # Cache expiry untuk query
    
    # ==================== Reflexion Settings ====================
    REFLEXION_MAX_ITERATIONS: int = 2  # Max self-critique iterations
    REFLEXION_MIN_QUALITY_SCORE: float = 8.0  # Min score untuk accept response (raised untuk higher quality)
    
    # ==================== Agent Settings ====================
    AGENT_TEMPERATURE: float = 0.3
    AGENT_MAX_TOKENS: int = 2000
    AGENT_TIMEOUT_SECONDS: int = 30

    # ==================== Logging ====================
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "ai_learning_insight.log"
    
    # ==================== Rate Limiting ====================
    RATE_LIMIT_CALLS: int = 10  # Max calls per time window
    RATE_LIMIT_WINDOW_SECONDS: int = 60

settings = Settings()

class Config:
    DB_HOST = os.getenv("DB_HOST_PROD")
    DB_DIALECT = os.getenv("DB_DIALECT_PROD")
    DB_USERNAME = os.getenv("DB_USERNAME_PROD")
    DB_PASSWORD = urllib.parse.quote(os.getenv("DB_PASSWORD_PROD"))
    DB_NAME = os.getenv("DB_NAME_PROD")
    
    SQLALCHEMY_DATABASE_URI = f"{DB_DIALECT}+asyncpg://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class DatabaseConfig:
    """Centralized database configuration"""

    HOST = os.getenv("DB_HOST_PROD")
    DATABASE = os.getenv("DB_NAME_PROD")
    USER = os.getenv("DB_USERNAME_PROD")
    PASSWORD = urllib.parse.quote(os.getenv("DB_PASSWORD_PROD"))
    PORT = int(os.getenv("DB_PORT_PROD", 5432))
    
    @classmethod
    def get_connection_params(cls):
        """Get psycopg2 connection parameters"""
        return {
            'host': cls.HOST,
            'database': cls.DATABASE,
            'user': cls.USER,
            'password': cls.PASSWORD,
            'port': cls.PORT
        }

__all__ = ['settings']