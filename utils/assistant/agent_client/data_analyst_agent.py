from langchain_core.tools import tool
from acp_sdk.client import Client

from app.config import settings
from colorama import Fore
from datetime import datetime
import traceback
import json

@tool
async def call_data_analyst_agent(
    user_id: str,
    analysis_type: str = "comprehensive",
    session_id: str = None
) -> str:
    """
    Call Data Analyst Agent untuk deep pattern analysis dari Cognee memory.
    
    Provides:
    - Learning pattern analysis (success/failure rates)
    - Engagement metrics and trends
    - Risk factor identification
    - Growth opportunity discovery
    - Data-driven insights
    
    Analysis Types:
    - "comprehensive": Full analysis of all patterns
    - "performance": Focus on success rates and outcomes
    - "engagement": Focus on activity and participation patterns
    - "risk": Focus on areas of concern and improvement needs
    
    Args:
        user_id (str): User to analyze
        analysis_type (str): Type of analysis to perform
        session_id (str): Optional session identifier
    
    Returns:
        str: JSON-formatted analysis with insights and recommendations
    """
    try:
        if session_id is None:
            session_id = f"analyst_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{Fore.LIGHTMAGENTA_EX}{'='*80}{Fore.RESET}")
        print(f"{Fore.LIGHTMAGENTA_EX} CALLING: Data Analyst Agent{Fore.RESET}")
        print(f"{Fore.LIGHTMAGENTA_EX}{'='*80}{Fore.RESET}")
        print(f"{Fore.LIGHTYELLOW_EX}User ID: {user_id}{Fore.RESET}")
        print(f"{Fore.LIGHTYELLOW_EX}Analysis: {analysis_type}{Fore.RESET}\n")
        
        request_data = json.dumps({
            "user_id": user_id,
            "analysis_type": analysis_type,
            "session_id": session_id
        }, ensure_ascii=False)
        
        async with Client(base_url=settings.AGENT_DATA_ANALYST_ENDPOINT) as client:
            stream = client.run_stream(
                agent="data_analyst_agent",
                input=request_data,
            )
            
            results = []
            async for chunk in stream:
                results.append(chunk)
                print(f"{Fore.LIGHTBLUE_EX}[STREAM] Analyzing...{Fore.RESET}", end='\r')
            
            if results:
                final_result = results[-1]
                
                if hasattr(final_result, 'output') and final_result.output:
                    content = final_result.output[0].parts[0].content
                elif hasattr(final_result, 'content'):
                    content = final_result.content
                else:
                    content = str(final_result)
                
                print(f"\n{Fore.LIGHTGREEN_EX}✓ Analysis complete{Fore.RESET}\n")
                return content
            else:
                return json.dumps({"error": "No data available from analyst"})
    
    except Exception as e:
        error_msg = f"Analysis Error: {str(e)}"
        print(f"{Fore.LIGHTRED_EX}{error_msg}{Fore.RESET}")
        return json.dumps({"error": error_msg})