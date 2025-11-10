from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routes.api_router import router
from datetime import datetime
import asyncio
import json
import os

from utils.assistant.schema import StreamAgentSchema
from utils.assistant.connection_manager import manager
from services.ai_assistant import rag_system
from utils.logger import logger
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

@ai_apps.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    user_uuid = None
    connection_id = None
    
    try:
        await websocket.accept()
        
        await websocket.send_text(json.dumps({
            "type": "connected",
            "output": "Connected to server. Please register with user_uuid.",
            "timestamp": datetime.now().isoformat()
        }))
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
            except json.JSONDecodeError as e:
                
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "output": f"Invalid JSON format: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }))

                continue
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break
            
            message_type = message.get("type", "")

            if message_type == "register":
                """
                /*
                    |--------------------------------------------------------------------------
                    | if message_type == "register"
                    |--------------------------------------------------------------------------
                    | Mengecek apakah pesan yang diterima bertipe "register". 
                    | Jika ya, maka sistem akan mengambil `user_uuid` dari pesan.
                    | Jika `user_uuid` tidak ada, akan dikirim error.
                    | Jika ada, user diregistrasikan ke `manager.connect`, lalu dikonfirmasi
                    | ke client bahwa user berhasil register dengan menyertakan connection_id.
                */
                """

                user_uuid = message.get("user_uuid")
                if not user_uuid:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "output": "user_uuid is required for registration",
                        "timestamp": datetime.now().isoformat()
                    }))
                    continue
                
                connection_id = await manager.connect(websocket, user_uuid)
                
                await websocket.send_text(json.dumps({
                    "type": "registered",
                    "output": f"User {user_uuid} registered successfully",
                    "user_uuid": user_uuid,
                    "connection_id": connection_id,
                    "timestamp": datetime.now().isoformat()
                }))

            elif message_type == "chat":
                """
                /*
                    |--------------------------------------------------------------------------
                    | if message_type == "chat"
                    |--------------------------------------------------------------------------
                    | Mengecek apakah pesan yang diterima bertipe "chat".
                    | Jika iya, maka akan diproses untuk interaksi chat setelah validasi.
                */
                """
                if not user_uuid:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "output": "Please register first with user_uuid",
                        "timestamp": datetime.now().isoformat()
                    }))
                    continue
                
                input_text = message.get("input", "").strip()
                
                if not input_text:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "output": "input field is required and cannot be empty",
                        "timestamp": datetime.now().isoformat()
                    }))
                    continue
                
                try:
                    data_wrapper = StreamAgentSchema(
                        input=input_text,
                    )

                    asyncio.create_task(rag_system.generate_response_streaming(user_uuid, data_wrapper))
                    
                except Exception as validation_error:
                    logger.error(f"Schema validation error: {validation_error}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "output": f"Invalid input format: {str(validation_error)}",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
            elif message_type == "get_user_info":
                """
                /*
                    |--------------------------------------------------------------------------
                    | if message_type == "get_user_info"
                    |--------------------------------------------------------------------------
                    | Mengecek apakah pesan yang diterima bertipe "get_user_info".
                    | Jika iya, sistem akan memberikan informasi apakah user sudah register,
                */
                """

                if not user_uuid:
                    await websocket.send_text(json.dumps({
                        "type": "user_info",
                        "output": {"registered": False, "message": "User not registered"},
                        "timestamp": datetime.now().isoformat()
                    }))
                    continue
                
                await websocket.send_text(json.dumps({
                    "type": "user_info",
                    "output": {
                        "registered": True,
                        "user_uuid": user_uuid,
                        "connection_id": connection_id,
                    },
                    "timestamp": datetime.now().isoformat()
                }))

            elif message_type == "ping":
                """
                /*
                    |--------------------------------------------------------------------------
                    | if message_type == "ping"
                    |--------------------------------------------------------------------------
                    | Mengecek apakah pesan yang diterima bertipe "ping".
                    | Jika iya, balas dengan pesan "pong" untuk memastikan koneksi hidup.
                */
                """

                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "output": f"Unknown message type: {message_type}",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user: {user_uuid}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_uuid}: {e}")
    finally:
        if connection_id and user_uuid:
            manager.disconnect(connection_id, user_uuid)
        if user_uuid and user_uuid in rag_system.user_processing:
            del rag_system.user_processing[user_uuid]