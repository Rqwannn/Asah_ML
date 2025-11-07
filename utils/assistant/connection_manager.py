from datetime import datetime
from utils.logger import logger
from typing import Any, Dict
import json
import time

from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}  # user_uuid -> websocket_id
    
    async def connect(self, websocket: WebSocket, user_uuid: str):
        """Connect a user and handle existing connections properly"""
        connection_id = f"{user_uuid}_{datetime.now().timestamp()}"
        
        if user_uuid in self.user_sessions:
            old_connection_id = self.user_sessions[user_uuid]
            if old_connection_id in self.active_connections:
                try:
                    old_websocket = self.active_connections[old_connection_id]
                    await old_websocket.send_text(json.dumps({
                        "type": "connection_replaced",
                        "output": "Connection replaced by new session",
                        "timestamp": datetime.now().isoformat()
                    }))
                    await old_websocket.close(code=1000)
                except Exception as e:
                    logger.warning(f"Error closing old connection: {e}")
                finally:
                    del self.active_connections[old_connection_id]
        
        self.active_connections[connection_id] = websocket
        self.user_sessions[user_uuid] = connection_id
        logger.info(f"User {user_uuid} connected with ID: {connection_id}")
        return connection_id
    
    def disconnect(self, connection_id: str, user_uuid: str = None):
        """Clean disconnect of connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if user_uuid and user_uuid in self.user_sessions:
            if self.user_sessions[user_uuid] == connection_id:
                del self.user_sessions[user_uuid]
        
        logger.info(f"Connection {connection_id} disconnected")
    
    async def send_message(self, message: dict, user_uuid: str):
        
        """Send message with proper error handling"""

        if user_uuid in self.user_sessions:
            connection_id = self.user_sessions[user_uuid]
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]
                try:
                    await websocket.send_text(json.dumps(message))
                    return True
                except Exception as e:
                    logger.error(f"Error sending message to {user_uuid}: {e}")
                    self.disconnect(connection_id, user_uuid)
                    return False
        return False
    
    async def send_final_streaming(self, user_uuid: str, final_result):

        if user_uuid not in self.user_sessions:
            logger.warning(f"User {user_uuid} not found in sessions")
            return False            
        
        success = await self.send_message(final_result, user_uuid)

        if not success:
            logger.error(f"Failed to send streaming message to {user_uuid}")
            return False
        
        return True
    
    async def send_streaming_message(self, start_time, text: str, user_uuid: str, chat_history: Any, message_type: str = "stream"):
        """Send text word by word for streaming effect with better error handling"""
        if user_uuid not in self.user_sessions:
            logger.warning(f"User {user_uuid} not found in sessions")
            return False
        
        if message_type == "response_stream":
        
            for texts in text:
                end_time = time.perf_counter()

                execution_time = end_time - start_time

                message = {
                    "type": message_type,
                    "output": texts,
                    "Execution_Time": execution_time
                }
                
                success = await self.send_message(message, user_uuid)

                if not success:
                    logger.error(f"Failed to send streaming message to {user_uuid}")
                    return False
        else :
            end_time = time.perf_counter()

            execution_time = end_time - start_time

            message = {
                "type": message_type,
                "output": text,
                "Execution_Time": execution_time
            }
            
            success = await self.send_message(message, user_uuid)

            if not success:
                logger.error(f"Failed to send streaming message to {user_uuid}")
                return False

        
        return True

manager = ConnectionManager()