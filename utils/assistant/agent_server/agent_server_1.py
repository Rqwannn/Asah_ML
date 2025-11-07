from collections.abc import AsyncGenerator

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from langgraph.prebuilt import create_react_agent
from mcp import StdioServerParameters

# from utils.assistant.tools.mcp_utils.vectordb_pengukuran import (
#     init_tools
# )

from utils.logger import logger
from app.config import settings
import json
import os

server = Server()

server_parameters = StdioServerParameters(
    command="python",
    args=["-m", "utils.assistant.tools.vectordb_pengukuran"],
    env=None,
)

mcp_agent = None
mcp_tools = None

async def init_mcp_on_start(server_parameters):
    global mcp_agent, mcp_tools
    mcp_agent, mcp_tools = await init_tools(server_parameters)

@server.agent()
async def pengukuran_agent(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    Enhanced health agent dengan ReAct reasoning dan multiple tools.
    """
    
    try:

        # sepertinya disini gunakan astream_events

        if isinstance(input[0].parts[0].content, str):
            data = json.loads(input[0].parts[0].content) 
        else:
            data = input[0].parts[0].content  

        user_input = data['task']
        parse_config = data['configurable']

        config = {"configurable": parse_config}

        result = await mcp_agent.ainvoke(
            {"input": user_input},
            config
        )
        
        final_answer = result.get("output", "Tidak dapat memproses permintaan.")
        intermediate_steps = result.get("intermediate_steps", [])
        
        logger.info(f"ReAct steps executed: {len(intermediate_steps)}")
        
        for i, step in enumerate(intermediate_steps):
            logger.info(f"Step {i+1}: {step[0].tool if hasattr(step[0], 'tool') else 'Unknown'}")
        
        yield Message(parts=[MessagePart(content=final_answer)])
        
    except Exception as e:
        logger.error(f"agent pengukuran error: {str(e)}")

        fallback_response = (
            "Maaf, sistem sedang mengalami gangguan. "
            "kamu diizinkan untuk memberikan jawaban berdasarkan pengetahuan umum yang kamu miliki. "
            "Pastikan jawaban tetap relevan, jelas, dan informatif. "
            "Untuk pengukuran baju, gunakan panduan ukuran standar atau kunjungi toko langsung. "
        )
        
        yield Message(parts=[MessagePart(content=fallback_response)])

if __name__ == "__main__":
    import asyncio
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
    asyncio.run(init_mcp_on_start(server_parameters))
    server.run(host="0.0.0.0", port=4040)