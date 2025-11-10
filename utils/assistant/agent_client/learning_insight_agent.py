from langchain_core.tools import tool
from acp_sdk.client import Client

from app.config import settings
from colorama import Fore
from datetime import datetime
import traceback
import json

from utils.assistant.agent_client.helper.setup import config_context

@tool
async def call_learning_insight_agent(
    task: str, 
    user_id: str = "anonymous",
    session_id: str = None
) -> str:
    """
    Call main Learning Insight Agent dengan Cognee + Reflexion + Advanced Orchestration.
    
    This agent provides:
    - Personalized learning recommendations based on Cognee memory
    - Self-critiqued responses (Reflexion with 5-criteria evaluation)
    - Pattern-aware suggestions from learning history
    - Adaptive strategies based on user's success/failure patterns
    
    Best for:
    - Learning advice and study strategies
    - Personalized recommendations
    - Overcoming learning obstacles
    - Skill development guidance
    
    Args:
        task (str): Learning query or request
        user_id (str): User identifier for Cognee memory isolation
        session_id (str): Optional session ID for grouping related interactions
    
    Returns:
        str: High-quality personalized recommendation (min 7.0/10 quality score)
    """

    try:
        config = config_context.get({})
        
        if session_id is None:
            session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config['configurable']['user_id'] = user_id
        config['configurable']['session_id'] = session_id
        
        print(f"\n{Fore.LIGHTCYAN_EX}{'='*80}{Fore.RESET}")
        print(f"{Fore.LIGHTCYAN_EX}🎓 CALLING: Learning Insight Agent{Fore.RESET}")
        print(f"{Fore.LIGHTCYAN_EX}{'='*80}{Fore.RESET}")
        print(f"{Fore.LIGHTYELLOW_EX}User ID: {user_id}{Fore.RESET}")
        print(f"{Fore.LIGHTYELLOW_EX}Session: {session_id}{Fore.RESET}")
        print(f"{Fore.LIGHTYELLOW_EX}Task: {task[:100]}...{Fore.RESET}\n")
        
        task_json = json.dumps({
            "task": task,
            "query": task,
            "configurable": config["configurable"]
        }, ensure_ascii=False)
        
        async with Client(base_url=settings.AGENT_LEARNING_INSIGHT_ENDPOINT) as client:
            stream = client.run_stream(
                agent="learning_insight_agent",
                input=task_json,
            )
            
            results = []
            async for chunk in stream:
                results.append(chunk)
                # Stream progress
                if hasattr(chunk, 'content'):
                    print(f"{Fore.LIGHTBLUE_EX}[STREAM] Receiving...{Fore.RESET}", end='\r')
            
            if results:
                final_result = results[-1]
                
                if hasattr(final_result, 'output') and final_result.output:
                    content = final_result.output[0].parts[0].content
                elif hasattr(final_result, 'content'):
                    content = final_result.content
                else:
                    content = str(final_result)
                
                print(f"\n{Fore.LIGHTGREEN_EX}✓ Response received ({len(content)} chars){Fore.RESET}\n")
                return content
            else:
                return "No response received from Learning Insight Agent"
    
    except ConnectionError as e:
        error_msg = f"Connection Error: Cannot connect to {settings.AGENT_LEARNING_INSIGHT_ENDPOINT}. Ensure server is running."
        print(f"{Fore.LIGHTRED_EX}{error_msg}{Fore.RESET}")
        return error_msg
    
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"{Fore.LIGHTRED_EX}{error_msg}{Fore.RESET}")
        return error_msg