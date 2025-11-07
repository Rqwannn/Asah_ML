from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import text, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import Config
from typing import AsyncGenerator

async_engine = create_async_engine(url=Config.SQLALCHEMY_DATABASE_URI, echo=True)

async def init_db():
    """Create the database tables"""
    async with async_engine.begin() as conn:
        from app.models.document import Document

        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to provide the session object"""
    async_session = sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session