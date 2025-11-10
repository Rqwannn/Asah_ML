from langchain_core.tools import tool
from acp_sdk.client import Client

from app.config import settings
from colorama import Fore
import json

@tool
async def call_reflexion_demo(
    query: str,
    user_id: str = "demo_user"
) -> str:
    """
    Demo Reflexion process untuk showcase self-critique mechanism.
    
    Shows:
    - Initial response generation
    - Multi-criteria evaluation (5 dimensions)
    - Iterative refinement process
    - Quality score progression
    - Final high-quality output
    
    Perfect for:
    - Understanding how Reflexion works
    - Seeing quality improvement in action
    - Learning about evaluation criteria
    - Testing system capabilities
    
    Args:
        query (str): Sample learning query
        user_id (str): Demo user identifier
    
    Returns:
        str: Detailed demonstration of Reflexion process with metrics
    """
    try:
        print(f"\n{Fore.YELLOW}{'='*80}{Fore.RESET}")
        print(f"{Fore.YELLOW} CALLING: Reflexion Demo Agent{Fore.RESET}")
        print(f"{Fore.YELLOW}{'='*80}{Fore.RESET}")
        print(f"{Fore.LIGHTYELLOW_EX}Query: {query[:100]}...{Fore.RESET}\n")
        
        request_data = json.dumps({
            "query": query,
            "user_id": user_id
        }, ensure_ascii=False)
        
        async with Client(base_url=settings.AGENT_REFLEXION_ENDPOINT) as client:
            stream = client.run_stream(
                agent="reflexion_demo_agent",
                input=request_data,
            )
            
            results = []
            async for chunk in stream:
                results.append(chunk)
            
            if results:
                final_result = results[-1]
                
                if hasattr(final_result, 'output') and final_result.output:
                    content = final_result.output[0].parts[0].content
                elif hasattr(final_result, 'content'):
                    content = final_result.content
                else:
                    content = str(final_result)
                
                print(f"\n{Fore.LIGHTGREEN_EX}✓ Demo complete{Fore.RESET}\n")
                return content
            else:
                return " No demo output received"
    
    except Exception as e:
        error_msg = f" Demo Error: {str(e)}"
        print(f"{Fore.LIGHTRED_EX}{error_msg}{Fore.RESET}")
        return error_msg