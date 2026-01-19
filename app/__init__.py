from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from routes.api_router import router
from dotenv import load_dotenv
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("server is starting")

    load_dotenv()

    os.environ['GOOGLE_API_KEY']
    os.environ['OPENAI_API_KEY']
    os.environ['LANCEDB_API_KEY']
    os.environ['COHERE_API_KEY']

    os.environ["LANGSMITH_TRACING"]
    os.environ["LANGCHAIN_PROJECT"]
    os.environ["LANGCHAIN_ENDPOINT"]
    os.environ["LANGCHAIN_API_KEY"]

    yield
    print("server is shutting down")


apps = FastAPI(
    title="AI Pengukuran",
    version="0.1.0",
    description="Untuk keperluan pengukuran baju",
    lifespan=lifespan,
)

apps.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

apps.include_router(router, tags=["agent"])