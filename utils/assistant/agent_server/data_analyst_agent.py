from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from collections.abc import AsyncGenerator

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server

from datetime import datetime
import json
import os

from utils.assistant.agent_server.helper.cognee_configuration import (
    get_or_create_cognee_tools
)
from utils.logger import logger
from app.config import settings

server = Server()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini", # atau "gpt-4o-mini"
    temperature=0.3,
    max_tokens=2000,
)

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.3,
#     max_tokens=2000,
# )

@server.agent()
async def data_analyst_agent(
    input: list[Message],
    context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    Data Analyst Agent - Deep dive into learning patterns dari Cognee
    """
    try:
        if isinstance(input[0].parts[0].content, str):
            data = json.loads(input[0].parts[0].content)
        else:
            data = input[0].parts[0].content
        
        user_id = data.get('user_id', 'anonymous')
        analysis_type = data.get('analysis_type', 'comprehensive')
        session_id = data.get('session_id', f"analyst_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        logger.info(f"[DATA ANALYST AGENT] Analyzing patterns for user {user_id}")
        
        # Get Cognee tools
        cognee_tools = await get_or_create_cognee_tools(session_id)
        
        # Comprehensive pattern search
        queries = [
            f"all learning activities and outcomes for user {user_id}",
            f"success rate and performance metrics for user {user_id}",
            f"learning velocity and engagement patterns for user {user_id}",
            f"areas of struggle and improvement opportunities for user {user_id}"
        ]
        
        search_tasks = [
            cognee_tools['search'].ainvoke({"query": q}) 
            for q in queries
        ]
        
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Aggregate analysis
        analysis = {
            "user_id": user_id,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "data_sources": {
                "learning_activities": results[0] if not isinstance(results[0], Exception) else "No data",
                "success_metrics": results[1] if not isinstance(results[1], Exception) else "No data",
                "engagement_patterns": results[2] if not isinstance(results[2], Exception) else "No data",
                "improvement_areas": results[3] if not isinstance(results[3], Exception) else "No data"
            },
            "summary": "Pattern analysis completed",
            "data_quality": {
                "sources_found": sum(1 for r in results if not isinstance(r, Exception) and r),
                "total_queries": len(queries)
            }
        }
        
        # Generate insights menggunakan LLM
        insight_prompt = f"""Analisis pola pembelajaran berikut dan berikan wawasan berbasis data:

        {json.dumps(analysis, ensure_ascii=False, indent=2)}

        Berikan:
        1. Temuan utama (3 - 5 poin)
        2. Tren dari waktu ke waktu (jika data tersedia)
        3. Faktor risiko (jika ada)
        4. Peluang untuk pengembangan

        Analisis:"""

        
        insights = await model.ainvoke([HumanMessage(content=insight_prompt)])
        analysis["insights"] = insights.content
        
        response = json.dumps(analysis, ensure_ascii=False, indent=2)
        yield Message(parts=[MessagePart(content=response)])
        
    except Exception as e:
        logger.error(f"[DATA ANALYST AGENT] Error: {str(e)}")
        
        error_response = json.dumps({
            "error": str(e),
            "message": "Data analysis failed, but system is operational"
        }, ensure_ascii=False)

        yield Message(parts=[MessagePart(content=error_response)])


if __name__ == "__main__":
    import asyncio
    
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
    
    print("\nConfiguration:")
    print(f"  - Model: Gemini 2.0 Flash Exp")
    print(f"  - Vector DB: LanceDB (local dev)")
    print(f"  - Graph DB: NetworkX (local dev)")
    print(f"  - Reflexion Iterations: {settings.REFLEXION_MAX_ITERATIONS}")
    print(f"  - Min Quality Score: {settings.REFLEXION_MIN_QUALITY_SCORE}")
    
    print("\n" + "=" * 80)
    print("Server starting on http://0.0.0.0:4041")
    print("=" * 80 + "\n")
    
    server.run(host="0.0.0.0", port=4041)