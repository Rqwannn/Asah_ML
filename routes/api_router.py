from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, Body
from pinecone import Pinecone
from http import HTTPStatus
from typing import List, Union

from dotenv import load_dotenv
import os

"""
/*
    |--------------------------------------------------------------------------
    | In the bottom line for document operation
    |--------------------------------------------------------------------------
*/
"""

router = APIRouter(prefix="/api")

@router.post("/add_document", status_code=HTTPStatus.CREATED)
async def add_document():
    pass