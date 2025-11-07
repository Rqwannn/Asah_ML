import psycopg2
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import logging
from config import DatabaseConfig, PineconeConfig
from pinecone import Pinecone

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Centralized database connection management"""
    
    @staticmethod
    @contextmanager
    def get_connection():
        """Context manager for database connections with auto-cleanup"""
        conn = None
        try:
            conn = psycopg2.connect(**DatabaseConfig.get_connection_params())
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    @contextmanager 
    def get_cursor(conn):
        """Context manager for cursors with auto-cleanup"""
        cursor = None
        try:
            cursor = conn.cursor()
            yield cursor
        finally:
            if cursor:
                cursor.close()


class DocumentRepository:
    """Repository pattern for document operations"""
    
    @staticmethod
    def save_documents(data: List[dict], namespace: str) -> bool:
        """Save multiple documents with transaction safety"""
        try:
            with DatabaseManager.get_connection() as conn:
                with DatabaseManager.get_cursor(conn) as cursor:
                    query = """
                        INSERT INTO pinecone_documents (id, filename, namespace) 
                        VALUES (%s, %s, %s) 
                        ON CONFLICT (id) DO NOTHING
                    """
                    
                    # Batch insert for better performance
                    cursor.executemany(query, [
                        (item["id"], item["filename"], namespace) 
                        for item in data
                    ])
                    
                    conn.commit()
                    logger.info(f"Saved {len(data)} documents to namespace: {namespace}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
            return False
    
    @staticmethod
    def get_documents(namespace: str, page: int, page_size: int) -> List[str]:
        """Get paginated documents"""
        try:
            offset = (page - 1) * page_size
            
            with DatabaseManager.get_connection() as conn:
                with DatabaseManager.get_cursor(conn) as cursor:
                    query = """
                        SELECT filename 
                        FROM pinecone_documents 
                        WHERE namespace = %s 
                        ORDER BY id 
                        LIMIT %s OFFSET %s
                    """
                    
                    cursor.execute(query, (namespace, page_size, offset))
                    documents = [row[0] for row in cursor.fetchall()]
                    
                    logger.info(f"Retrieved {len(documents)} documents from namespace: {namespace}")
                    return documents
                    
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return []
    
    @staticmethod
    def count_documents(namespace: str) -> int:
        """Count total documents in namespace"""
        try:
            with DatabaseManager.get_connection() as conn:
                with DatabaseManager.get_cursor(conn) as cursor:
                    query = "SELECT COUNT(*) FROM pinecone_documents WHERE namespace = %s"
                    cursor.execute(query, (namespace,))
                    count = cursor.fetchone()[0]
                    
                    logger.info(f"Found {count} documents in namespace: {namespace}")
                    return count
                    
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
    
    @staticmethod
    def delete_documents(filenames: List[str]) -> bool:
        """Delete documents by filenames"""
        try:
            with DatabaseManager.get_connection() as conn:
                with DatabaseManager.get_cursor(conn) as cursor:
                    query = "DELETE FROM pinecone_documents WHERE filename = ANY(%s)"
                    cursor.execute(query, (filenames,))
                    
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                    logger.info(f"Deleted {deleted_count} documents")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False


class PineconeService:
    """Service for Pinecone operations"""
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Pinecone client"""
        if self._client is None:
            self._client = Pinecone(api_key=PineconeConfig.API_KEY)
        return self._client
    
    def fetch_data_by_page_range(
        self,
        filenames: List[str],
        namespace: str,
        page_number: int,
        chunk_page_size: int = 200
    ) -> List[Dict[str, Any]]:
        """Fetch data from Pinecone with pagination"""
        try:
            index = self.client.Index(namespace)
            all_data = []
            
            page_start = (page_number - 1) * chunk_page_size
            page_end = page_start + chunk_page_size
            
            for filename in filenames:
                filter_query = {
                    "$and": [
                        {"filename": filename},
                        {"page": {"$gte": page_start, "$lt": page_end}}
                    ]
                }
                
                response = index.query(
                    vector=[0.0] * PineconeConfig.VECTOR_DIMENSION,
                    top_k=1000,
                    include_metadata=True,
                    namespace=namespace,
                    filter=filter_query,
                    filter_only=True
                )
                
                for match in response.get("matches", []):
                    all_data.append({
                        "id": match["id"],
                        "metadata": match.get("metadata", {})
                    })
            
            logger.info(f"Fetched {len(all_data)} items from Pinecone")
            return all_data
            
        except Exception as e:
            logger.error(f"Pinecone fetch error: {e}")
            return []


document_repo = DocumentRepository()
pinecone_service = PineconeService()