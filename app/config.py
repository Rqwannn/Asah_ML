from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os
import urllib.parse

load_dotenv()

class Settings(BaseSettings):
    """
    App settings loaded from environment variables or .env file
    """

    # API Keys
    PINECONE_API_KEY: str | None = None
    COHERE_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    LANGCHAIN_API_KEY: str | None = None

    # LangSmith / LangChain
    LANGSMITH_TRACING: bool = True
    LANGCHAIN_PROJECT: str = "learnaly-tica"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # Pinecone Vector DB
    PINECONE_VECTOR_DB: str = "LearnalyTica"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Agent Endpoint
    AGENT_PENGUKURAN_ENDPOINT: str | None = None

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

class PineconeConfig:
    """Pinecone configuration"""
    API_KEY = os.getenv("PINECONE_API_KEY")
    VECTOR_DIMENSION = int(os.getenv("PINECONE_DIMENSION", 1536))