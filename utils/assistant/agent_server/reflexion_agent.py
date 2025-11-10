# Agent Server - Advanced Implementation dengan Cognee, Reflexion, dan ACP

from collections.abc import AsyncGenerator
import json
import os
from datetime import datetime

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.logger import logger
from utils.assistant.agent_server.helper.reflexion_agent_config import ReflexionAgent
from utils.assistant.agent_server.helper.cognee_configuration import (
    get_or_create_cognee_tools
)
from app.config import settings

server = Server()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=2000,
)

@server.agent()
async def reflexion_demo_agent(
    input: list[Message],
    context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    Demo agent untuk showcase Reflexion process dengan detailed logging
    """
    try:
        if isinstance(input[0].parts[0].content, str):
            data = json.loads(input[0].parts[0].content)
        else:
            data = input[0].parts[0].content
        
        query = data.get('query', '')
        user_id = data.get('user_id', 'demo_user')
        session_id = f"reflexion_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"[REFLEXION DEMO] Starting demo for: {query}")
        
        # Get Cognee tools
        cognee_tools = await get_or_create_cognee_tools(session_id)
        
        # Create Reflexion agent
        reflexion_agent = ReflexionAgent(
            model=model,
            cognee_add_tool=cognee_tools['add'],
            cognee_search_tool=cognee_tools['search']
        )
        
        # Demo context
        demo_context = {
            "user_id": user_id,
            "session_id": session_id,
            "learning_style": "pembelajar visual dan praktikal (belajar dengan praktik langsung)",
            "past_performance": "Kesulitan dalam konsep abstrak, unggul dalam penerapan praktis"
        }
        
        # Run reflexion
        result = await reflexion_agent.generate_with_reflection(
            query=query,
            user_context=demo_context,
            session_id=session_id
        )
        
        # Detailed response showing the process
        demo_response = f"""**DEMONSTRASI PROSES REFLEKSI**

        **Kueri Asli:** {query}

        **Proses:**
        ✓ Respons awal dihasilkan  
        ✓ Evaluasi multi-kriteria dilakukan  
        ✓ Respons disempurnakan sebanyak {result['iterations']} kali  
        ✓ Kualitas meningkat sebesar {result['improvement']:.1f} poin

        **Skor Kualitas Akhir:** {result['final_score']:.1f}/10

        **Rincian Kriteria (Iterasi Terakhir):**
        {json.dumps(result['critique_history'][-1]['critique']['scores'], indent=2)}

        **Rekomendasi Akhir:**
        {result['final_response']}

        ---
        **Wawasan Proses:**
        - Jumlah iterasi: {result['iterations']}
        - Perkembangan skor: {' → '.join([f"{c['score']:.1f}" for c in result['critique_history']])}
        - Peningkatan utama: {', '.join(result['critique_history'][-1]['critique'].get('strengths', []))}

        *Ini menunjukkan bagaimana Reflexion memastikan respons yang berkualitas tinggi dan personal melalui proses kritik diri.*
        """
        
        yield Message(parts=[MessagePart(content=demo_response)])
        
    except Exception as e:
        logger.error(f"[REFLEXION DEMO] Error: {str(e)}")
        yield Message(parts=[MessagePart(content=f"Demo error: {str(e)}")])


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
    print("Server starting on http://0.0.0.0:4042")
    print("=" * 80 + "\n")
    
    server.run(host="0.0.0.0", port=4042)