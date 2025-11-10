from contextvars import ContextVar
from langgraph.checkpoint.memory import MemorySaver

config_context: ContextVar[dict] = ContextVar('config_context', default={})

checkpointer = MemorySaver()