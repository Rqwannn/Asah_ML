from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routes.api_router import router
from datetime import datetime
import asyncio
import os

from services.ai_assistant import rag_system
from dotenv import load_dotenv
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("server is starting")

    load_dotenv()

    os.environ['GOOGLE_API_KEY'] = settings.GOOGLE_API_KEY
    os.environ['PINECONE_API_KEY']
    os.environ['COHERE_API_KEY']

    os.environ["LANGSMITH_TRACING"]
    os.environ["LANGCHAIN_PROJECT"]
    os.environ["LANGCHAIN_ENDPOINT"]
    os.environ["LANGCHAIN_API_KEY"]

    yield
    print("server is shutting down")

ai_apps = FastAPI(
    title="Stream Real Time WebSocket", 
    debug=True,
    lifespan=lifespan
)

ai_apps.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_apps.include_router(router, tags=["agent"])