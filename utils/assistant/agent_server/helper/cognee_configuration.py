# utils/cognee_config.py atau tempat initialize_cognee() berada

from typing import Dict, Any
from contextvars import ContextVar
from app.config import settings
from utils.logger import logger
from pathlib import Path
import os

from cognee_integration_langgraph import get_sessionized_cognee_tools

current_user_context: ContextVar[Dict[str, str]] = ContextVar(
    'current_user_context', 
    default={}
)

async def initialize_cognee():
    """
    Inisialisasi Cognee dengan LanceDB Cloud endpoint
    """
    try:
        import cognee
        
        LANCEDB_URI = "db://learnalytica-txf2rg"
        LANCEDB_API_KEY = settings.LANCEDB_API_KEY  # Pastikan ada di .env
        
        BASE_DIR = Path(__file__).resolve().parent.parent  
        COGNEE_DATA_DIR = BASE_DIR / "data" / "cognee_storage"
        COGNEE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set LLM config
        await cognee.config.set_llm_config({
            "provider": "google",
            "model": "gemini-2.0-flash",
            "api_key": settings.GOOGLE_API_KEY,
            "temperature": 0.3,
        })
        
        # Set Vector DB - CLOUD LANCEDB
        await cognee.config.set_vector_db_config({
            "provider": "lancedb",
            "uri": LANCEDB_URI, 
            "api_key": LANCEDB_API_KEY, 
            "region": "us-east-1" 
        })
        
        await cognee.config.set_graph_db_config({
            "provider": "networkx",
            "path": str(COGNEE_DATA_DIR / "graph.pkl")
        })
        
        logger.info("✓ Cognee berhasil diinisialisasi")
        logger.info(f"  - LLM: Gemini 2.0 Flash")
        logger.info(f"  - Vector DB: LanceDB Cloud ({LANCEDB_URI})")
        logger.info(f"  - Graph DB: NetworkX (local)")
        logger.info(f"  - Graph Path: {COGNEE_DATA_DIR / 'graph.pkl'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Gagal menginisialisasi Cognee: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

cognee_initialized = False
cognee_tools_cache = {}

async def get_or_create_cognee_tools(session_id: str) -> Dict[str, Any]:
    """
    Get or create sessionized Cognee tools
    """
    global cognee_tools_cache, cognee_initialized
    
    if not cognee_initialized:
        success = await initialize_cognee()
        if not success:
            raise RuntimeError("Failed to initialize Cognee")
        cognee_initialized = True
    
    if session_id in cognee_tools_cache:
        logger.info(f"[COGNEE] Using cached tools for session {session_id}")
        return cognee_tools_cache[session_id]
    
    logger.info(f"[COGNEE] Creating new sessionized tools for session {session_id}")
    
    tools = get_sessionized_cognee_tools(
        session_id=session_id,
        add_tool_name="cognee_add_memory",
        add_tool_description="Tambahkan data pembelajaran, refleksi, dan interaksi ke memori permanen Cognee",
        search_tool_name="cognee_search_memory",
        search_tool_description="Cari memori Cognee untuk menemukan pola pembelajaran, riwayat, dan konteks relevan menggunakan kueri bahasa alami"
    )

    cognee_tools_cache[session_id] = {
        'add': tools[0],
        'search': tools[1],
        'raw_tools': tools
    }
    
    logger.info(f"[COGNEE] ✓ Created and cached tools for session {session_id}")
    return cognee_tools_cache[session_id]