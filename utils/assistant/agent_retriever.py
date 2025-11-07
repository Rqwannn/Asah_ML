from typing import List, Type, Dict, Any
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import time
import re

from langchain_core.tools import BaseTool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, PrivateAttr

from contextvars import ContextVar

from utils.logger import logger

class CustomRetriever:
    def __init__(self, vectorstore, cache_expiry_seconds=3600):
        self.vectorstore = vectorstore
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_expiry_seconds = cache_expiry_seconds
        
    def get_relevant_documents(self, query: str, limit: int = 3) -> List[Document]:
        self._clean_expired_cache()
        
        cache_key = f"{query}_{limit}"
        if cache_key in self.cache:
            logger.info(f"Cache hit for query: {query}")
            # Perbarui timestamp saat cache digunakan
            self.cache_timestamps[cache_key] = time.time()
            return self.cache[cache_key]
        
        try:
            results = self.vectorstore.similarity_search(query, k=limit)
            self.cache[cache_key] = results
            self.cache_timestamps[cache_key] = time.time()
            return results
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return []
            
    def _clean_expired_cache(self):
        """Membersihkan cache yang sudah melewati waktu kadaluwarsa"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_expiry_seconds:
                expired_keys.append(key)
                
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            del self.cache_timestamps[key]
            
        if expired_keys:
            logger.info(f"Cleaned {len(expired_keys)} expired cache entries")

class CustomReActOutputParser(ReActSingleInputOutputParser):
    def parse(self, text):
        
        # try:
        #     return super().parse(text)
        # except Exception:
        try:
            lower_text = text.lower()

            if "final answer:" in lower_text:
                index = lower_text.find("final answer:")
                final_answer = text[index + len("final answer:"):].strip()

                if "action:" in lower_text and "action input:" in lower_text:
                    action_match = re.search(r"(?i)\baction\b\s*:\s*(\w+)", text)
                    input_match = re.search(
                        r"(?i)\baction[_\s]*input\b\s*:\s*(.*?)(?:$|\baction\b\s*:|\bobservation\b\s*:|\bfinal\s+answer\b\s*:)",
                        text,
                        re.DOTALL
                    )
                    
                    if action_match and input_match:
                        action = action_match.group(1).strip()
                        action_input_raw = input_match.group(1).strip()

                        action_input = {"query": action_input_raw}

                        return AgentAction(
                            tool=action,
                            tool_input=action_input,
                            log=text
                        )

                return AgentFinish(
                    return_values={"output": final_answer},
                    log=text
                )

            elif "action:" in lower_text and "action input:" in lower_text:
                action_match = re.search(r"(?i)\baction\b\s*:\s*(\w+)", text)
                input_match = re.search(
                    r"(?i)\baction[_\s]*input\b\s*:\s*(.*?)(?:$|\baction\b\s*:|\bobservation\b\s*:|\bfinal\s+answer\b\s*:)",
                    text,
                    re.DOTALL
                )
                
                if action_match and input_match:
                    action = action_match.group(1).strip()
                    action_input_raw = input_match.group(1).strip()

                    action_input = {"query": action_input_raw}

                    return AgentAction(
                        tool=action,
                        tool_input=action_input,
                        log=text
                    )
            
            return AgentFinish(
                return_values={"output": text},
                log=text
            )
        except Exception as e:
            logger.error(f"Error in custom parser: {str(e)}")
            return AgentFinish(
                return_values={"output": "Maaf, terjadi kesalahan teknis. Terima kasih sudah bertanya!"},
                log=text
            )
            
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def add_messages(self, messages: List[BaseMessage]) -> None:
        if datetime.now(timezone.utc) - self.last_updated > timedelta(hours=1):
            self.messages = []

        self.messages.extend(messages)
        self.last_updated = datetime.now(timezone.utc)

        if len(self.messages) > 3:
            self.messages = self.messages[-3:]

    def clear(self) -> None:
        self.messages = []
        self.last_updated = datetime.now(timezone.utc)

class InputSchema(BaseModel):
    input_data: str
    
class LimitedAsyncTool(BaseTool):
    name: str = "TemplateDocumentTool"
    description: str = (
        "Alat ini digunakan untuk membuat dokumen sesuai template. "
        "Gunakan jika pengguna meminta pembuatan dokumen template atau menyebut 'template'/'laporan'/'buat dokumen'."
    )

    args_schema: Type[BaseModel] = InputSchema

    _tool_func: Any = PrivateAttr()
    _max_calls: int = PrivateAttr()
    _time_window: timedelta = PrivateAttr()
    _user_call_times: Dict[str, list] = PrivateAttr()
    _context: ContextVar[Dict[str, Any]] = PrivateAttr()

    def __init__(self, tool_func, max_calls=6, time_window_sec=30, context: ContextVar = None):
        super().__init__()
        self._tool_func = tool_func
        self._max_calls = max_calls
        self._time_window = timedelta(seconds=time_window_sec)
        self._user_call_times = defaultdict(list)
        self._context = context or ContextVar("user_context")

    async def _arun(self, input_data: str) -> str:
        context = self._context.get({})
        user_id = context.get("user_id", "anonymous")
        now = datetime.now()

        timestamps = self._user_call_times[user_id]
        
        # Filter: hanya simpan yang masih dalam window 30 detik

        timestamps = [ts for ts in timestamps if now - ts < self._time_window]
        self._user_call_times[user_id] = timestamps

        if len(timestamps) >= self._max_calls:
            logger.warning(f"[LIMIT] User {user_id} mencapai batas {self._max_calls} call dalam 30 detik.")
            return (
                f"Alat '{self.name}' telah digunakan sebanyak {self._max_calls} kali dalam 30 detik terakhir.\n"
                f"Silakan gunakan informasi yang telah disediakan dari hasil sebelumnya untuk menyusun tanggapan Anda."
            )

        self._user_call_times[user_id].append(now)
        return await self._tool_func.ainvoke({"input_data": input_data})

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Gunakan agent async agar tool async dapat dipanggil.")
    
store = {}

def get_session_history(
    user_id: str, conversation_id: str
) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryHistory()
    return store[(user_id, conversation_id)]