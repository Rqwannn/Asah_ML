from sqlmodel import SQLModel, Field,Column
import sqlalchemy.dialects.postgresql as pg
from uuid import UUID, uuid4
from datetime import datetime

class Document(SQLModel, table = True):
    """
    This class represents a book in the database
    """
    __tablename__ = 'pinecone_documents'
    uid:UUID = Field(
        sa_column=Column(pg.UUID ,primary_key=True,
        unique=True, default=uuid4)
    )
    filename:str
    namespace:str
    
    created_at: datetime = Field(sa_column=Column(pg.TIMESTAMP, default=datetime.now))

    def __repr__(self) -> str:
        return f"Document => {self.title}"