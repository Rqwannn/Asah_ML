from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader

import lancedb
from tempfile import NamedTemporaryFile
from fastapi import UploadFile
import uuid
from pathlib import Path

from helper.ocr import process_with_ocr
from app.database.lance_manager import lancedb_manager

import os
from typing import List, Union, Optional
from dotenv import load_dotenv
from math import ceil

class PDFService:
    
    def __init__(self):
        load_dotenv()

        self.gemini_api_key = os.environ['GOOGLE_API_KEY']
        self.cohere_api_key = os.environ['COHERE_API_KEY']
        
        self.lancedb_uri = "db://learnalytica-txf2rg"
        self.lancedb_api_key = os.environ.get('LANCEDB_API_KEY')
        
        self.default_table_name = "learnalytica-academy"
        
        if not self.lancedb_api_key:
            raise ValueError("LANCEDB_API_KEY must be set in environment variables")
        
        # Use centralized LanceDB manager
        self.lancedb = lancedb_manager
        self.db = self.lancedb.client
        
        print("=" * 60)
        print("LanceDB Cloud Connection Initialized")
        print(f"  URI: {self.lancedb_uri}")
        print(f"  Default Table: {self.default_table_name}")
        print(f"  Region: us-east-1")
        print("=" * 60)

    def _get_table_name(self, db_name: Optional[str] = None) -> str:
        """Get table name, default to learnalytica-academy if not provided"""
        return db_name if db_name else self.default_table_name

    def _get_or_create_table(self, table_name: str):
        """Get existing table atau create new table di LanceDB Cloud"""
        try:
            existing_tables = self.db.table_names()
            
            if table_name in existing_tables:
                table = self.db.open_table(table_name)
                print(f"Opened existing table: {table_name}")
                return table
            else:
                print(f"Table {table_name} doesn't exist yet, will be created")
                return None
        except Exception as e:
            print(f"Error checking table: {e}")
            return None

    async def get_document_agent(
        self, 
        db_name: Optional[str] = None, 
        page: int = 1, 
        page_size: int = 10
    ):
        """
        Get paginated documents from LanceDB metadata table
        """
        table_name = self._get_table_name(db_name)
        
        # Get from LanceDB metadata (no more PostgreSQL)
        documents = self.lancedb.get_document_metadata(table_name, page, page_size)
        total = self.lancedb.count_documents(table_name)

        return {
            "table_name": table_name,
            "page": page,
            "page_size": page_size,
            "total_data": total,
            "total_pages": ceil(total / page_size) if total > 0 else 0,
            "results": documents
        }

    async def get_detail_document_agent(
        self, 
        filename: Union[str, List[str]], 
        page: int = 0,
        db_name: Optional[str] = None
    ):
        """Get document details from LanceDB Cloud by filename and page"""
        table_name = self._get_table_name(db_name)
        filenames = [filename] if isinstance(filename, str) else filename
        
        # Use lancedb_manager method
        results = self.lancedb.fetch_data_by_page_range(
            filenames=filenames,
            namespace=table_name,
            page_number=page + 1,  # Convert 0-indexed to 1-indexed
            chunk_page_size=200
        )
        
        return {
            "table_name": table_name,
            "filename": filenames,
            "page": page,
            "total_results": len(results),
            "results": results
        }

    async def add_documents(
        self, 
        files: List[UploadFile], 
        db_name: Optional[str] = None
    ):
        """
        Add documents to LanceDB Cloud WITHOUT OCR
        """
        table_name = self._get_table_name(db_name)
        results = []
        temp_files = []
        all_documents = []

        try:
            print(f"Processing documents for table: {table_name}")
            
            # Step 1: Save uploaded files temporarily
            for file in files:
                suffix = file.filename.split('.')[-1].lower()

                if suffix not in ["pdf", "doc", "docx"]:
                    results.append({
                        "filename": file.filename,
                        "status": "Unsupported file type"
                    })
                    continue

                with NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp_path = tmp.name
                    temp_files.append({
                        "path": tmp_path,
                        "filename": file.filename,
                        "suffix": suffix,
                        "file_size": len(content)
                    })

            # Step 2: Load documents
            for temp_file in temp_files:
                try:
                    if temp_file["suffix"] == "pdf":
                        loader = PyPDFLoader(temp_file["path"])
                    elif temp_file["suffix"] in ["doc", "docx"]:
                        loader = UnstructuredWordDocumentLoader(temp_file["path"])

                    docs = loader.load()
                    
                    # Add metadata
                    for i, doc in enumerate(docs):
                        doc.metadata["filename"] = temp_file["filename"]
                        doc.metadata["source"] = temp_file["filename"]
                        doc.metadata["document_type"] = "text"
                        doc.metadata["page_number"] = i + 1
                        doc.metadata["table_name"] = table_name

                    all_documents.extend(docs)

                    results.append({
                        "filename": temp_file["filename"],
                        "status": "Loaded successfully",
                        "document_count": len(docs),
                        "file_size": temp_file["file_size"]
                    })
                except Exception as e:
                    results.append({
                        "filename": temp_file["filename"],
                        "status": f"Error loading document: {str(e)}"
                    })

            # Step 3: Process and add to LanceDB Cloud
            if all_documents:
                print(f"Uploading {len(all_documents)} documents to LanceDB Cloud...")
                
                embeddings = CohereEmbeddings(
                    model="embed-v4.0",
                    cohere_api_key=self.cohere_api_key
                )
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=100
                )
                chunked_docs = text_splitter.split_documents(all_documents)
                
                print(f"Split into {len(chunked_docs)} chunks")

                for doc in chunked_docs:
                    doc.metadata["id"] = str(uuid.uuid4())

                try:
                    table = self._get_or_create_table(table_name)
                    
                    if table is None:
                        print(f"Creating new table '{table_name}' in LanceDB Cloud...")
                        vectorstore = LanceDB.from_documents(
                            documents=chunked_docs,
                            embedding=embeddings,
                            connection=self.db,
                            table_name=table_name
                        )
                        print(f"Created table: {table_name}")
                    else:
                        print(f"Adding to existing table '{table_name}'...")
                        vectorstore = LanceDB(
                            connection=self.db,
                            table_name=table_name,
                            embedding=embeddings
                        )
                        vectorstore.add_documents(chunked_docs)
                        print(f"Added {len(chunked_docs)} chunks to {table_name}")
                    
                    for i, result in enumerate(results):
                        if "document_count" in result:
                            results[i]["status"] = "Document uploaded and indexed successfully"
                            results[i]["chunks_created"] = len([
                                d for d in chunked_docs 
                                if d.metadata.get("filename") == result["filename"]
                            ])
                            results[i]["table_name"] = table_name
                            results[i]["storage"] = "LanceDB Cloud (All data)"
                
                except Exception as e:
                    print(f"Error adding to LanceDB Cloud: {e}")
                    import traceback
                    print(traceback.format_exc())
                    for i, result in enumerate(results):
                        if "document_count" in result:
                            results[i]["status"] = f"Error indexing: {str(e)}"

            # Step 4: Save metadata to LanceDB (replaces PostgreSQL)
            for result in results:
                if result.get("status", "").startswith("Document uploaded"):
                    doc_record = {
                        "id": str(uuid.uuid4()),
                        "filename": result["filename"],
                        "file_size": result.get("file_size", 0),
                        "document_type": "text"
                    }
                    self.lancedb.save_document_metadata([doc_record], namespace=table_name)

            return results

        finally:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file["path"])
                except:
                    pass

    async def add_documents_with_ocr(
        self, 
        files: List[UploadFile], 
        db_name: Optional[str] = None
    ):
        """
        Add documents to LanceDB Cloud WITH OCR support
        """
        table_name = self._get_table_name(db_name)
        results = []
        temp_files = []
        all_documents = []

        try:
            print(f"Processing documents with OCR for table: {table_name}")
            
            for file in files:
                suffix = file.filename.split('.')[-1].lower()

                if suffix not in ["pdf", "doc", "docx", "jpg", "jpeg", "png", "tiff", "tif"]:
                    results.append({
                        "filename": file.filename,
                        "status": "Unsupported file type"
                    })
                    continue

                with NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp_path = tmp.name
                    temp_files.append({
                        "path": tmp_path,
                        "filename": file.filename,
                        "suffix": suffix,
                        "file_size": len(content)
                    })

            for temp_file in temp_files:
                try:
                    docs = []
                    suffix = temp_file["suffix"]
                    
                    if suffix == "pdf":
                        try:
                            loader = PyPDFLoader(temp_file["path"])
                            docs = loader.load()
                            
                            if not docs or all(not doc.page_content.strip() for doc in docs):
                                print(f"Scanned PDF detected: {temp_file['filename']}, using OCR")
                                docs = process_with_ocr(temp_file["path"])
                        except Exception as e:
                            print(f"PDF loading failed, trying OCR: {e}")
                            docs = process_with_ocr(temp_file["path"])
                            
                    elif suffix in ["doc", "docx"]:
                        loader = UnstructuredWordDocumentLoader(temp_file["path"])
                        docs = loader.load()
                    
                    elif suffix in ["jpg", "jpeg", "png", "tiff", "tif"]:
                        docs = process_with_ocr(temp_file["path"])
                    
                    for i, doc in enumerate(docs):
                        doc.metadata["filename"] = temp_file["filename"]
                        doc.metadata["source"] = temp_file["filename"]
                        doc.metadata["document_type"] = "scanned" if suffix in ["jpg", "jpeg", "png", "tiff", "tif"] else "text"
                        doc.metadata["page_number"] = i + 1
                        doc.metadata["table_name"] = table_name
                    
                    all_documents.extend(docs)
                    
                    results.append({
                        "filename": temp_file["filename"],
                        "status": "Loaded successfully",
                        "document_count": len(docs),
                        "file_size": temp_file["file_size"],
                        "ocr_used": suffix in ["jpg", "jpeg", "png", "tiff", "tif"]
                    })
                    
                except Exception as e:
                    results.append({
                        "filename": temp_file["filename"],
                        "status": f"Error loading document: {str(e)}"
                    })

            if all_documents:
                print(f"Uploading {len(all_documents)} documents to LanceDB Cloud with OCR...")
                
                embeddings = CohereEmbeddings(
                    model="embed-v4.0",
                    cohere_api_key=self.cohere_api_key
                )
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=100
                )
                chunked_docs = text_splitter.split_documents(all_documents)
                
                print(f"Split into {len(chunked_docs)} chunks")

                for doc in chunked_docs:
                    doc.metadata["id"] = str(uuid.uuid4())

                try:
                    table = self._get_or_create_table(table_name)
                    
                    if table is None:
                        print(f"Creating new table '{table_name}' in LanceDB Cloud...")
                        vectorstore = LanceDB.from_documents(
                            documents=chunked_docs,
                            embedding=embeddings,
                            connection=self.db,
                            table_name=table_name
                        )
                        print(f"Created table: {table_name}")
                    else:
                        print(f"Adding to existing table '{table_name}'...")
                        vectorstore = LanceDB(
                            connection=self.db,
                            table_name=table_name,
                            embedding=embeddings
                        )
                        vectorstore.add_documents(chunked_docs)
                        print(f"Added {len(chunked_docs)} chunks")
                    
                    for i, result in enumerate(results):
                        if "document_count" in result:
                            results[i]["status"] = "Document uploaded and indexed successfully"
                            results[i]["chunks_created"] = len([
                                d for d in chunked_docs 
                                if d.metadata.get("filename") == result["filename"]
                            ])
                            results[i]["table_name"] = table_name
                            results[i]["storage"] = "LanceDB Cloud (All data)"
                
                except Exception as e:
                    print(f"Error adding to LanceDB Cloud: {e}")
                    import traceback
                    print(traceback.format_exc())
                    for i, result in enumerate(results):
                        if "document_count" in result:
                            results[i]["status"] = f"Error indexing: {str(e)}"

            # Save metadata to LanceDB
            for result in results:
                if result.get("status", "").startswith("Document uploaded"):
                    doc_record = {
                        "id": str(uuid.uuid4()),
                        "filename": result["filename"],
                        "file_size": result.get("file_size", 0),
                        "document_type": result.get("document_type", "scanned" if result.get("ocr_used") else "text")
                    }
                    self.lancedb.save_document_metadata([doc_record], namespace=table_name)

            return results
        
        finally:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file["path"])
                except:
                    pass
    
    async def process_single_document(
        self, 
        file: UploadFile, 
        type_doc: str,
        db_name: Optional[str] = None
    ):
        """Process single document"""
        table_name = self._get_table_name(db_name)