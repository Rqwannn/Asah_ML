from acp_sdk.client import Client
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver 
from langchain.agents import create_agent
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

from utils.assistant.schema import State
from app.config import settings

import json
import traceback
from colorama import Fore 
from contextvars import ContextVar

config_context: ContextVar[dict] = ContextVar('config_context', default={})

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    model_kwargs={"streaming": True},
    temperature=0.2,
    max_tokens=2500,
    top_p=0.85,
    top_k=20,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
)

checkpointer = InMemorySaver()

@tool
async def run_pengukuran_workflow(task: str) -> str:
    """
    Menjalankan workflow pengukuran dengan memanggil agent `pengukuran_agent`.

    Args:
        task (str): Input task untuk pengukuran dalam format string.

    Returns:
        str: Output teks hasil dari agent pengukuran.
    """
    try:

        config = config_context.get({})
        
        print(f"Trying to call pengukuran_agent with task: {task}\n")

        task_json = {
            "task": task,
            "configurable": config["configurable"]
        }

        task_json = json.dumps(task_json, ensure_ascii=False)

        async with Client(base_url=settings.AGENT_PENGUKURAN_ENDPOINT) as client:
            # async for event in client.run_stream(

            stream = client.run_stream(
                agent="pengukuran_agent", 
                input=task_json,
            )
            
            results = []
            async for chunk in stream:
                results.append(chunk)
                print(Fore.LIGHTYELLOW_EX + f"Received chunk : {chunk}" + Fore.RESET)
                print("")
            
            if results:
                final_result = results[-1]  
                
                if hasattr(final_result, 'output') and final_result.output:
                    content = final_result.output[0].parts[0].content
                elif hasattr(final_result, 'content'):
                    content = final_result.content
                else:
                    content = str(final_result)
                    
                print(Fore.LIGHTGREEN_EX + f"Tool Success: {content}" + Fore.RESET)
                return content
            else:
                error_msg = "No results received from pengukuran_agent"
                print(Fore.LIGHTRED_EX + error_msg + Fore.RESET)
                return error_msg
            
    except ConnectionError as e:
        error_msg = f"Connection Error: Tidak dapat terhubung ke service di localhost:4040. Pastikan service berjalan. Detail: {str(e)}"
        print(Fore.LIGHTRED_EX + error_msg + Fore.RESET)
        return error_msg
        
    except Exception as e:
        error_msg = f"Tool Error: {str(e)}\n{traceback.format_exc()}"
        print(Fore.LIGHTRED_EX + error_msg + Fore.RESET)
        return error_msg

# Tool untuk testing koneksi
@tool 
async def test_connection() -> str:
    """
    Test koneksi ke service pengukuran.
    
    Returns:
        str: Status koneksi
    """
    try:
        async with Client(base_url="http://localhost:4040") as client:
            return "Koneksi ke service berhasil"
    except Exception as e:
        return f"Koneksi gagal: {str(e)}"
    
async def execute(config, message):  
    print(Fore.LIGHTGREEN_EX + "Starting agent execution..." + Fore.RESET)

    config_context.set(config)

    agent = create_agent(
        model=model,
        tools=[run_pengukuran_workflow, test_connection],
        checkpointer=checkpointer,
        state_schema=State,
        prompt="""
                Kamu adalah asisten yang membantu dalam ...
               """
    )

    input_message = {"messages": [{"role": "user", "content": message['message']}]}
    
    try:
        config_thread = {"configurable": {"thread_id": "vin-calc-1"}}
        result = await agent.ainvoke(input_message, config_thread)
        
        final_response = result["messages"][-1].content
        print(Fore.LIGHTCYAN_EX + f"Final Response: {final_response}" + Fore.RESET)
        
        return final_response
        
    except Exception as e:
        error_response = f"Agent Error: {str(e)}\n{traceback.format_exc()}"
        print(Fore.LIGHTRED_EX + error_response + Fore.RESET)
        return f"Maaf, terjadi kesalahan dalam memproses permintaan: {str(e)}"

# Usage example:
# {
# "type": "chat", 
# "input": "Lingkar dada: 90 cm Lingkar pinggang: 84 cm Panjang lengan: 59 cm Lebar bahu: 45 cm Panjang baju: 71 cm"
# }