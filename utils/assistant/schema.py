from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel
from typing import Annotated
from operator import add

class InputSchema(BaseModel):
    input_data: str

class StreamAgentSchema(BaseModel):
    input: str

class State(AgentState):
    result: Annotated[str, add]
    config: Annotated[dict, add] = {}