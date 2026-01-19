from langchain_google_genai import ChatGoogleGenerativeAI
from collections.abc import AsyncGenerator

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server

from datetime import datetime
import json
import os

from utils.assistant.agent_server.helper.cognee_configuration import (
    current_user_context,
    get_or_create_cognee_tools
)
from utils.assistant.agent_server.helper.multi_agent_orchestrator import MultiAgentOrchestrator
from utils.logger import logger
from app.config import settings

server = Server()

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.3,
#     max_tokens=2000,
# )

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini", # atau "gpt-4o-mini"
    temperature=0.3,
    max_tokens=2000,
)

@server.agent()
async def learning_insight_agent(
    input: list[Message],
    context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    Main Learning Insight Agent dengan:
    - Cognee persistent memory
    - Advanced Reflexion
    - Multi-agent orchestration
    """

    try:
        # Parse input
        if isinstance(input[0].parts[0].content, str):
            data = json.loads(input[0].parts[0].content)
        else:
            data = input[0].parts[0].content
        
        user_query = data.get('task', data.get('query', ''))
        config = data.get('configurable', {})
        user_id = config.get('user_id', 'anonymous')
        session_id = config.get('session_id', f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        logger.info(f"[LEARNING INSIGHT AGENT] Processing for user {user_id}, session {session_id}")
        
        # Set context
        current_user_context.set({
            'user_id': user_id,
            'session_id': session_id
        })
        
        # Get Cognee tools
        cognee_tools = await get_or_create_cognee_tools(session_id)
        
        # Create orchestrator
        orchestrator = MultiAgentOrchestrator(
            model=model,
            cognee_tools=cognee_tools
        )
        
        # Process request
        result = await orchestrator.process_learning_request(
            user_id=user_id,
            query=user_query,
            session_id=session_id,
            config=config
        )
        
        # Format respons berdasarkan jenisnya
        if result['response_type'] == 'learning_recommendation':
            final_response = f"""**Wawasan Pembelajaran AI - Rekomendasi Pribadi**
            (Skor Kualitas: {result['quality_score']:.1f}/10, Disempurnakan melalui {result['reflexion_iterations']} iterasi)

            {result['recommendation']}

            ---
            *Rekomendasi ini dibuat khusus untuk Anda berdasarkan pola pembelajaran yang tersimpan di Memori Cognee.*
            """
        elif result['response_type'] == 'pattern_analysis':
            final_response = f"""**Analisis Pola Pembelajaran**

            **Pola Anda:**
            {result['patterns']}

            **Wawasan Utama:**
            {result['insights']}

            **Rekomendasi:**
            {result['recommendations']}
            """
        elif result['response_type'] == 'motivation':
            final_response = result['message']

        elif result['response_type'] == 'comprehensive':
            final_response = f"""**Analisis Pembelajaran Komprehensif**

            {result['synthesized_insight']}

            ---
            **Analisis Pola:**
            {json.dumps(result['pattern_analysis'], ensure_ascii=False, indent=2)}

            ---
            **Rekomendasi Pribadi:**
            (Kualitas: {result['personalized_recommendation']['quality_score']:.1f}/10)

            {result['personalized_recommendation']['recommendation']}
            """
        else:
            final_response = json.dumps(result, ensure_ascii=False, indent=2)
        
        # Cognify data (batch process untuk efficiency)
        try:
            import cognee
            await cognee.cognify()
            logger.info("[COGNEE] ✓ Cognified new data")
        except Exception as e:
            logger.warning(f"[COGNEE] Cognify warning: {e}")
        
        yield Message(parts=[MessagePart(content=final_response)])
        
    except Exception as e:
        logger.error(f"[LEARNING INSIGHT AGENT] Error: {str(e)}", exc_info=True)
        
        fallback_response = f"""**Pemberitahuan Sistem**

        Saya mengalami kendala teknis, namun saya tetap dapat membantu berdasarkan prinsip umum pembelajaran:

        **Tips Belajar Cepat:**
        1. **Pecah Menjadi Bagian Kecil**: Bagi topik kompleks menjadi bagian yang lebih kecil dan mudah dipahami
        2. **Latihan Aktif**: Belajarlah dengan praktik langsung, bukan hanya membaca
        3. **Pengulangan Bersela (Spaced Repetition)**: Tinjau kembali materi dengan jeda waktu yang semakin panjang
        4. **Ajarkan ke Orang Lain**: Menjelaskan konsep membantu memperkuat pemahaman Anda
        5. **Hubungkan dengan Pengetahuan Sebelumnya**: Kaitkan informasi baru dengan apa yang sudah Anda ketahui

        **Jika Anda merasa buntu:**
        - Istirahatlah sejenak dan kembali dengan pikiran segar  
        - Ajukan pertanyaan spesifik daripada mencoba memahami semuanya sekaligus  
        - Gunakan berbagai sumber (video, artikel, proyek praktis)  
        - Jangan bandingkan kemajuan Anda dengan orang lain — setiap orang memiliki ritme belajarnya sendiri  

        Terus semangat! Setiap ahli pernah menjadi pemula. 🚀

        ---
        *Detail kesalahan: {str(e)}*
        
        """

        yield Message(parts=[MessagePart(content=fallback_response)])

if __name__ == "__main__":
    import asyncio
    
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

    print("\nComponents:")
    print("  ✓ Cognee Integration: Persistent semantic memory with knowledge graphs")
    print("  ✓ Advanced Reflexion: Multi-criteria self-critique (5 dimensions)")
    print("  ✓ Multi-Agent Orchestrator: Intelligent routing and parallel execution")
    print("  ✓ ACP Protocol: Seamless agent-to-agent communication")

    print("\nAvailable Agents:")
    print("  1. learning_insight_agent - Main AI Learning Mentor")
    print("  2. data_analyst_agent - Deep pattern analysis")
    print("  3. reflexion_demo_agent - Showcase reflexion process")
    
    print("\nConfiguration:")
    print(f"  - Model: Gemini 2.0 Flash Exp")
    print(f"  - Vector DB: LanceDB (local dev)")
    print(f"  - Graph DB: NetworkX (local dev)")
    print(f"  - Reflexion Iterations: {settings.REFLEXION_MAX_ITERATIONS}")
    print(f"  - Min Quality Score: {settings.REFLEXION_MIN_QUALITY_SCORE}")
    
    print("\n" + "=" * 80)
    print("Server starting on http://0.0.0.0:4040")
    print("=" * 80 + "\n")

    cognee_tools = get_or_create_cognee_tools(1)
    
    server.run(host="0.0.0.0", port=4040)