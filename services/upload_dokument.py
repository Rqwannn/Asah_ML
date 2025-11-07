from langchain_cohere import CohereEmbeddings

from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from langchain_core.outputs import Generation
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from tempfile import NamedTemporaryFile
from fastapi import UploadFile
import uuid

from helper.ocr import process_with_ocr
from helper.batching_file import batch_documents_by_size
from utils.schema import *
from utils.postgres import *

import os
from typing import List, Union
from dotenv import load_dotenv
from math import ceil

class PDFService:
    
    def __init__(self):
        load_dotenv()

        self.gemini_api_key = os.environ['GOOGLE_API_KEY']
        self.pinecone_api_key = os.environ['PINECONE_API_KEY']
        self.cohere_api_key = os.environ['COHERE_API_KEY']

        os.environ["LANGSMITH_TRACING"]
        os.environ["LANGCHAIN_PROJECT"]
        os.environ["LANGCHAIN_ENDPOINT"]
        os.environ["LANGCHAIN_API_KEY"]

    async def get_document_agent(self, db_name: str, page: int, page_size: int):
        total = count_documents_in_postgres(db_name)
        document = get_document_from_postgres(db_name, page, page_size)

        return {
            "page": page,
            "page_size": page_size,
            "total_data": total,
            "total_pages": ceil(total / page_size),
            "results": document
        }

    async def get_detail_document_agent(self, db_name: str, filename: Union[str, List[str]], page):
        filenames = [filename] if isinstance(filename, str) else filename
        results = fetch_data_from_pinecone_by_page_range(filenames, db_name, page_number=page)

        return {
            "filename": filenames,
            "page": page,
            "page_range": [(page) * 200, (page * 200)],
            "results": results
        }

    async def add_documents(self, files: List[UploadFile], db_name: str):
        results = []
        temp_files = []
        all_documents = []

        try:
            for file in files:
                suffix = file.filename.split('.')[-1].lower()

                if suffix not in ["pdf", "doc", "docx"]:
                    results.append({
                        "filename": file.filename,
                        "status": "Unsupported file type"
                    })
                    continue

                with NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                    tmp.write(await file.read())
                    tmp_path = tmp.name
                    temp_files.append({
                        "path": tmp_path,
                        "filename": file.filename,
                        "suffix": suffix
                    })

            for temp_file in temp_files:
                try:
                    if temp_file["suffix"] == "pdf":
                        loader = PyPDFLoader(temp_file["path"])
                    elif temp_file["suffix"] in ["doc", "docx"]:
                        loader = UnstructuredWordDocumentLoader(temp_file["path"])

                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["filename"] = temp_file["filename"]

                    all_documents.extend(docs)

                    results.append({
                        "filename": temp_file["filename"],
                        "status": "Loaded successfully",
                        "document_count": len(docs)
                    })
                except Exception as e:
                    results.append({
                        "filename": temp_file["filename"],
                        "status": f"Error loading document: {str(e)}"
                    })

            if all_documents:
                embeddings = CohereEmbeddings(model="embed-v4.0")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=100
                )

                chunked_docs = text_splitter.split_documents(all_documents)

                doc_ids = [str(uuid.uuid4()) for _ in chunked_docs]
                for doc, doc_id in zip(chunked_docs, doc_ids):
                    doc.metadata["ID"] = doc_id

                pinecone = Pinecone()
                if db_name not in pinecone.list_indexes().names():
                    pinecone.create_index(
                        name=db_name,
                        dimension=1536,
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )

                vectordb = PineconeVectorStore.from_existing_index(
                    index_name=db_name,
                    embedding=embeddings
                )

                batches = batch_documents_by_size(chunked_docs, doc_ids)

                for docs_batch, ids_batch in batches:
                    vectordb.add_documents(
                        documents=docs_batch,
                        ids=ids_batch,
                        namespace=db_name
                    )

                for i, result in enumerate(results):
                    if "document_count" in result:
                        results[i]["status"] = "Document uploaded and indexed successfully"

            for result in results:
                if result.get("status", "").startswith("Document uploaded"):
                    doc_record = {
                        "id": str(uuid.uuid4()),  # hanya satu ID acak per file
                        "filename": result["filename"]
                    }
                    save_document_to_postgres([doc_record], namespace=db_name)

            return results

        finally:
            # Hapus file sementara
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file["path"])
                except:
                    pass

    async def add_documents_with_ocr(self, files: List[UploadFile], db_name: str):

        """
        Process and add documents to Pinecone, with OCR support for scanned documents.
        Handles PDF, DOC, DOCX, JPG, JPEG, PNG, and TIFF files.
        """

        results = []
        temp_files = []
        all_documents = []

        try:
            for file in files:
                suffix = file.filename.split('.')[-1].lower()

                if suffix not in ["pdf", "doc", "docx", "jpg", "jpeg", "png", "tiff", "tif"]:
                    results.append({
                        "filename": file.filename,
                        "status": "Unsupported file type"
                    })
                    continue

                with NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                    tmp.write(await file.read())
                    tmp_path = tmp.name
                    temp_files.append({
                        "path": tmp_path,
                        "filename": file.filename,
                        "suffix": suffix
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
                                docs = process_with_ocr(temp_file["path"])
                        except Exception:
                            docs = process_with_ocr(temp_file["path"])
                            
                    elif suffix in ["doc", "docx"]:
                        loader = UnstructuredWordDocumentLoader(temp_file["path"])
                        docs = loader.load()
                    
                    elif suffix in ["jpg", "jpeg", "png", "tiff", "tif"]:
                        docs = process_with_ocr(temp_file["path"])
                    
                    for doc in docs:
                        doc.metadata["filename"] = temp_file["filename"]
                    
                    all_documents.extend(docs)
                    results.append({
                        "filename": temp_file["filename"],
                        "status": "Loaded successfully",
                        "document_count": len(docs)
                    })
                except Exception as e:
                    results.append({
                        "filename": temp_file["filename"],
                        "status": f"Error loading document: {str(e)}"
                    })

            if all_documents:
                embeddings = CohereEmbeddings(model="embed-v4.0")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=100
                )

                chunked_docs = text_splitter.split_documents(all_documents)

                doc_ids = [str(uuid.uuid4()) for _ in chunked_docs]

                for doc, doc_id in zip(chunked_docs, doc_ids):
                    doc.metadata["ID"] = doc_id

                pinecone = Pinecone()
                if db_name not in pinecone.list_indexes().names():
                    pinecone.create_index(
                        name=db_name,
                        dimension=1536,
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )

                vectordb = PineconeVectorStore.from_existing_index(index_name=db_name, embedding=embeddings)
                # vectordb.add_documents(documents=chunked_docs, ids=doc_ids, namespace=db_name)

                batches = batch_documents_by_size(chunked_docs, doc_ids)

                for docs_batch, ids_batch in batches:
                    vectordb.add_documents(
                        documents=docs_batch,
                        ids=ids_batch,
                        namespace=db_name
                    )

                for i, result in enumerate(results):
                    if "document_count" in result:
                        results[i]["status"] = "Document uploaded and indexed successfully"

            for result in results:
                if result.get("status", "").startswith("Document uploaded"):
                    doc_record = {
                        "id": str(uuid.uuid4()),  # hanya satu ID acak per file
                        "filename": result["filename"]
                    }
                    save_document_to_postgres([doc_record], namespace=db_name)

            return results
        
        finally:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file["path"])
                except:
                    pass
    
    async def process_single_document(self, file: UploadFile, db_name: str, type_doc: str):
        if type_doc == "scan":
            results = await self.add_documents_with_ocr([file], db_name)
        else:
            results = await self.add_documents([file], db_name)

        if results and len(results) > 0:
            return results[0]["status"]
        return "Error processing document"