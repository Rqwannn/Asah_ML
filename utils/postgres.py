from app.database.manager import document_repo, pinecone_service
from typing import List, Dict, Any

def save_document_to_postgres(data: List[dict], namespace: str) -> bool:
    return document_repo.save_documents(data, namespace)

def get_document_from_postgres(namespace: str, page: int, page_size: int) -> List[str]:
    return document_repo.get_documents(namespace, page, page_size)

def count_documents_in_postgres(namespace: str) -> int:
    return document_repo.count_documents(namespace)

def delete_document_from_postgres(filenames: List[str]) -> bool:
    return document_repo.delete_documents(filenames)

def fetch_data_from_pinecone_by_page_range(
    filenames: List[str],
    namespace: str, 
    page_number: int,
    chunk_page_size: int = 200
) -> List[Dict[str, Any]]:
    return pinecone_service.fetch_data_by_page_range(
        filenames, namespace, page_number, chunk_page_size
    )