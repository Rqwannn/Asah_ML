from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from typing import Annotated
from operator import add

class InputSchema(BaseModel):
    input_data: str

class StreamAgentSchema(BaseModel):
    input: str

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: Optional[str]
    session_id: Optional[str]