import lancedb
from typing import List, Dict, Any, Optional
import logging
from app.config import LanceDBConfig

logger = logging.getLogger(__name__)


class LanceDBManager:
    """
    Centralized LanceDB Cloud connection and operations
    All document metadata and vectors stored in LanceDB
    """
    
    def __init__(self):
        self._client = None
        self.default_table = LanceDBConfig.DEFAULT_TABLE
        self.metadata_table = "document_metadata"  
    
    @property
    def client(self):
        """Lazy initialization of LanceDB client"""
        if self._client is None:
            self._client = lancedb.connect(
                uri=LanceDBConfig.URI,
                api_key=LanceDBConfig.API_KEY,
                region=LanceDBConfig.REGION
            )
            logger.info(f"Connected to LanceDB Cloud: {LanceDBConfig.URI}")
        return self._client
    
    def _get_table_name(self, namespace: Optional[str] = None) -> str:
        """Get table name from namespace"""
        return namespace if namespace else self.default_table
    
    def _ensure_metadata_table(self):
        """
        Ensure metadata table exists
        This replaces PostgreSQL pinecone_documents table
        """
        try:
            existing_tables = self.client.table_names()
            
            if self.metadata_table not in existing_tables:
                import pyarrow as pa
                
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("filename", pa.string()),
                    pa.field("namespace", pa.string()),
                    pa.field("created_at", pa.string()),
                    pa.field("file_size", pa.int64()),
                    pa.field("document_type", pa.string()),
                ])
                
                self.client.create_table(
                    name=self.metadata_table,
                    schema=schema,
                    mode="create"
                )
                logger.info(f"Created metadata table: {self.metadata_table}")
        except Exception as e:
            logger.warning(f"Metadata table check: {e}")
    
    # ==================== METADATA OPERATIONS ====================
    
    def save_document_metadata(self, data: List[dict], namespace: str) -> bool:
        """
        Save document metadata to LanceDB
        Replaces: save_document_to_postgres
        """
        try:
            self._ensure_metadata_table()
            table = self.client.open_table(self.metadata_table)
            
            import pyarrow as pa
            from datetime import datetime
            
            records = []
            for item in data:
                records.append({
                    "id": item["id"],
                    "filename": item["filename"],
                    "namespace": namespace,
                    "created_at": datetime.now().isoformat(),
                    "file_size": item.get("file_size", 0),
                    "document_type": item.get("document_type", "unknown")
                })
            
            table.add(records)
            
            logger.info(f"Saved {len(data)} document metadata to {self.metadata_table}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save document metadata: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_document_metadata(
        self, 
        namespace: str, 
        page: int = 1, 
        page_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get paginated document metadata from LanceDB
        Replaces: get_document_from_postgres
        """
        try:
            table = self.client.open_table(self.metadata_table)
            
            offset = (page - 1) * page_size
            
            results = (
                table.search()
                .where(f"namespace = '{namespace}'")
                .limit(page_size)
                .to_list()
            )
            
            paginated = results[offset:offset + page_size]
            
            logger.info(f"Retrieved {len(paginated)} metadata from {namespace}")
            return paginated
            
        except Exception as e:
            logger.error(f"Failed to get document metadata: {e}")
            return []
    
    def count_documents(self, namespace: str) -> int:
        """
        Count documents in namespace
        Replaces: count_documents_in_postgres
        """
        try:
            table = self.client.open_table(self.metadata_table)
            
            results = (
                table.search()
                .where(f"namespace = '{namespace}'")
                .to_list()
            )
            
            count = len(results)
            logger.info(f"Found {count} documents in namespace: {namespace}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
    
    def delete_document_metadata(self, filenames: List[str]) -> bool: # PERLU DI REVISI
        """
        Delete document metadata by filenames
        Replaces: delete_document_from_postgres
        
        Note: LanceDB doesn't support direct delete by filter
        This is a simplified version - for production consider soft delete
        """
        try:
            logger.warning(f"Delete requested for {len(filenames)} files")
            logger.warning("LanceDB requires table rebuild for deletion. Consider soft delete.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document metadata: {e}")
            return False
    
    # ==================== VECTOR DATA OPERATIONS ====================
    
    def fetch_data_by_page_range(
        self,
        filenames: List[str],
        namespace: str,
        page_number: int,
        chunk_page_size: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Fetch vector data from LanceDB Cloud with pagination
        Replaces: fetch_data_from_pinecone_by_page_range
        """
        try:
            table_name = self._get_table_name(namespace)
            
            try:
                table = self.client.open_table(table_name)
            except Exception as e:
                logger.warning(f"Table {table_name} not found: {e}")
                return []
            
            page_start = (page_number - 1) * chunk_page_size
            page_end = page_start + chunk_page_size
            
            all_data = []
            
            for filename in filenames:
                where_clause = f"filename = '{filename}' AND page_number >= {page_start} AND page_number < {page_end}"
                
                try:
                    results = (
                        table.search()
                        .where(where_clause, prefilter=True)
                        .limit(1000)
                        .to_list()
                    )
                    
                    # Convert to compatible format
                    for row in results:
                        all_data.append({
                            "id": row.get("id", ""),
                            "metadata": {
                                "filename": row.get("filename", ""),
                                "page": row.get("page_number", 0),
                                "page_number": row.get("page_number", 0),
                                "source": row.get("source", ""),
                                "document_type": row.get("document_type", ""),
                                "table_name": row.get("table_name", ""),
                            }
                        })
                    
                except Exception as query_error:
                    logger.error(f"Query error for filename {filename}: {query_error}")
                    continue
            
            logger.info(f"Fetched {len(all_data)} items from LanceDB table: {table_name}")
            return all_data
            
        except Exception as e:
            logger.error(f"LanceDB fetch error: {e}")
            return []
    
    def get_table_stats(self, namespace: str) -> Dict[str, Any]:
        """Get statistics about a table"""
        try:
            table_name = self._get_table_name(namespace)
            table = self.client.open_table(table_name)
            
            return {
                "table_name": table_name,
                "total_rows": table.count_rows(),
                "schema": str(table.schema)
            }
        except Exception as e:
            logger.error(f"Failed to get table stats: {e}")
            return {"error": str(e)}
    
    def list_tables(self) -> List[str]:
        """List all available tables"""
        try:
            tables = self.client.table_names()
            logger.info(f"Found {len(tables)} tables in LanceDB")
            return tables
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []


lancedb_manager = LanceDBManager()