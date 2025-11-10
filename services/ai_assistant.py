from typing import List, Any, Dict
from dotenv import load_dotenv
from collections import deque
from datetime import datetime

import re
import time
import asyncio
import os

from utils.logger import logger
from utils.assistant.schema import StreamAgentSchema
from utils.assistant.connection_manager import manager
from utils.assistant.agent_client.agent_client import execute

class AgentService:
    def __init__(self):
        load_dotenv()
        
        self.user_processing = {}  

        self.is_sender_running = False
        self.word_queue: asyncio.Queue[str] = asyncio.Queue()

        self.gemini_api_key = os.environ['GOOGLE_API_KEY']
        self.pinecone_api_key = os.environ['PINECONE_API_KEY']
        self.cohere_api_key = os.environ['COHERE_API_KEY']

    def _serialize_chat_history(self, chat_history: List[Any]) -> List[Dict[str, Any]]:
        """Serialize chat history"""
        serialized = []
        
        for msg in chat_history:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                serialized.append({
                    "type": msg.type,
                    "content": msg.content
                })
            else:
                serialized.append({"type": "unknown", "content": str(msg)})
        
        return serialized
    
    async def stream_word_sender(self, manager, start_time, user_id):
        while True:
            word = await self.word_queue.get()
            if word is None:
                break

            await manager.send_streaming_message(start_time, word, user_id, [], "response_stream")
            await asyncio.sleep(0.03)

    def tokenize_content(self, text: str):
        return re.findall(r'\s+|[^\s]+', text)
    
    async def generate_agent(self, state: Any):

        config = {
            "configurable": {
                "user_id": state["user_id"],
                "conversation_id": "test_conversation",
            }
        }

        message = {
            "message": state["question"]
        }

        user_id = state['user_id']
        start_time = time.perf_counter()

        # Realtime Agent Streaming

        try:
            output = await asyncio.wait_for(execute(config, message), timeout=30)
        except asyncio.TimeoutError:
            await manager.send_streaming_message("Request timeout", user_id, "", "error")
            return

        await manager.send_streaming_message(start_time, output, user_id, [], message_type="response_stream")

    async def generate_response_streaming(self, user_id: Any, user_input: StreamAgentSchema):
        """Generate response using RAG with streaming and proper error handling"""
        
        if user_id in self.user_processing:
            await manager.send_message({
                "type": "error",
                "output": "Masih memproses pesan sebelumnya. Mohon tunggu...",
                "timestamp": datetime.now().isoformat()
            }, user_id)
            return
        
        self.user_processing[user_id] = True
        
        try:
            await manager.send_message({
                "type": "thinking",
                "output": "Sedang berpikir...",
                "timestamp": datetime.now().isoformat()
            }, user_id)
            
            doc_output_dir = os.path.join(os.getcwd(), "generated_documents")
            os.makedirs(doc_output_dir, exist_ok=True)

            try:

                initial_state = {
                    "user_id": user_id,
                    "conversation_id": "asdaslsadl",
                    "question": user_input.input,
                }

                await self.generate_agent(initial_state)
                
            except Exception as e:
                logger.error(f"Error creating agent or getting response: {e}")
                await manager.send_streaming_message(f"Maaf, terjadi kesalahan dalam memproses pertanyaan Anda: {str(e)}", user_id, "", "error")
                return
            
        except Exception as e:
            logger.error(f"Error generating response for user {user_id}: {e}")
            error_message = f"Maaf, terjadi kesalahan dalam memproses pertanyaan Anda: {str(e)}"
            await manager.send_streaming_message(error_message, user_id, "", "error")
        
        finally:
            # Always remove from processing list
            if user_id in self.user_processing:
                del self.user_processing[user_id]

rag_system = AgentService()