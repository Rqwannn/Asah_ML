from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, Body
from pinecone import Pinecone
from http import HTTPStatus
from typing import List, Union

from dotenv import load_dotenv
import os

from utils.schema import DeleteRequest
from services.upload_dokument import *
from services.classification_inference import ClassificationService
from utils.parsing_response import json_response
from utils.schema import ClassificationFeatures

router = APIRouter(prefix="/api")

@router.post("/measurement_analysis", status_code=HTTPStatus.CREATED)
async def stream_measurement_analysis(
    params: ClassificationFeatures = Body(...)
):
    try:
        result = await ClassificationService().analysis(params)

        final_response = json_response("CREAD_ACTION", 
                            {
                                "status": 201,
                                "message": "Berhasil melakukan analisis ukuran baju",
                                "result": result,
                            }
                        ), 201

        return final_response
    except HTTPException as he:
        # Jika sudah dilempar di service
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service gagal: {str(e)}")

"""
/*
    |--------------------------------------------------------------------------
    | In the bottom line for document operation
    |--------------------------------------------------------------------------
*/
"""

@router.post("/add_document", status_code=HTTPStatus.CREATED)
async def add_document(
    file: UploadFile = File(..., alias="file"), 
    type_doc: str = Form(...),
    task: str = Form("pengukuran")
    ):

    try:
        result = await PDFService().process_single_document(file, task, type_doc)
    
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/multiple_add_documents", status_code=HTTPStatus.CREATED)
async def add_documents(
    file: List[UploadFile] = File(..., alias="file"), 
    type_doc: str = Form(...),
    task: str = Form("pengukuran")
    ):

    try:
        if type_doc == "scan":
            print("\n=========== SCAN ===========")
            results = await PDFService().add_documents_with_ocr(file, task)
        else:
            results = await PDFService().add_documents(file, task)

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get_document", status_code=HTTPStatus.OK)
async def get_documents(
    page: int = Query(1, ge=1, description="Halaman ke-berapa"),
    task: str = Query("pengukuran", description="Nama task yang diminta (opsional)"),
):
    try:
        PAGE_SIZE = 15
        results = await PDFService().get_document_agent(task, page, PAGE_SIZE)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/detail_document", status_code=HTTPStatus.OK)
async def detail_documents(
    filename: Union[str, List[str]] = Query(..., description="Nama dokumen"),
    page: int = Query(1, ge=1, description="Nomor halaman dokumen (200 page per halaman API)"),
    task: str = Query("pengukuran", description="Nama task yang diminta (opsional)"),
):
    try:
        results = await PDFService().get_detail_document_agent(task, filename, page)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete_per_ids")
async def delete_items(
    request: DeleteRequest = Body(...),
):
    from time import sleep
    load_dotenv()
    INDEX_NAME = request.task
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    if INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(f"Index {INDEX_NAME} tidak ditemukan di Pinecone.")

    index = pc.Index(INDEX_NAME)

    filenames = [request.filename] if isinstance(request.filename, str) else request.filename

    filter_query = {"$or": [{"filename": name} for name in filenames]}

    all_deleted_ids = []

    while True:
        response = index.query(
            vector=[0.0]*1536,
            top_k=1000,
            include_metadata=True,
            namespace=INDEX_NAME,
            filter=filter_query,
            filter_only=True
        )

        matches = response.get("matches", [])
        ids_to_delete = [match["id"] for match in matches]

        if not ids_to_delete:
            break  

        index.delete(ids=ids_to_delete, namespace=INDEX_NAME)
        all_deleted_ids.extend(ids_to_delete)

        sleep(0.5) 

    delete_document_from_postgres(filenames)

    return {
        "status": "success",
        "filename": filenames,
        "deleted_vector_count": len(all_deleted_ids)
    }