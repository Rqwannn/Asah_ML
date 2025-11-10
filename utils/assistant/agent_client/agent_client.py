# Agent Client - Advanced Multi-Agent Communication via ACP

from acp_sdk.client import Client
from langchain_core.tools import tool
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
import asyncio
from colorama import Fore, Style
from typing import Dict, Any, List
from datetime import datetime

from utils.assistant.agent_client.helper.setup import checkpointer, config_context
from utils.assistant.agent_client.data_analyst_agent import call_data_analyst_agent
from utils.assistant.agent_client.learning_insight_agent import call_learning_insight_agent
from utils.assistant.agent_client.reflexion_agent import call_reflexion_demo

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    model_kwargs={"streaming": True},
    temperature=0.2,
    max_tokens=3000,
    top_p=0.85,
    top_k=20,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
)

@tool
async def multi_agent_comprehensive_analysis(
    user_id: str,
    query: str,
    include_patterns: bool = True,
    include_recommendations: bool = True,
    session_id: str = None
) -> str:
    """
    Execute comprehensive analysis menggunakan multiple agents secara parallel.
    
    Workflow:
    1. Data Analyst: Extract patterns dari Cognee
    2. Learning Insight: Generate recommendations dengan Reflexion
    3. Synthesis: Combine both untuk holistic insight
    
    Perfect for:
    - Complete learning assessment
    - Strategic planning
    - Progress reviews
    - Identifying blockers and opportunities
    
    Args:
        user_id (str): User to analyze
        query (str): Specific question or context
        include_patterns (bool): Include pattern analysis
        include_recommendations (bool): Include personalized recommendations
        session_id (str): Optional session grouping
    
    Returns:
        str: Comprehensive analysis combining multiple agent outputs
    """
    try:
        if session_id is None:
            session_id = f"comprehensive_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{Fore.LIGHTMAGENTA_EX}{'='*80}{Fore.RESET}")
        print(f"{Fore.LIGHTMAGENTA_EX} MULTI-AGENT COMPREHENSIVE ANALYSIS{Fore.RESET}")
        print(f"{Fore.LIGHTMAGENTA_EX}{'='*80}{Fore.RESET}")
        print(f"{Fore.LIGHTYELLOW_EX}User: {user_id}{Fore.RESET}")
        print(f"{Fore.LIGHTYELLOW_EX}Query: {query[:80]}...{Fore.RESET}")
        print(f"{Fore.LIGHTYELLOW_EX}Session: {session_id}{Fore.RESET}\n")
        
        tasks = []
        
        # Task 1: Pattern Analysis (if requested)
        if include_patterns:
            print(f"{Fore.LIGHTCYAN_EX}[1/2] Launching Data Analyst...{Fore.RESET}")
            tasks.append(call_data_analyst_agent(
                user_id=user_id,
                analysis_type="comprehensive",
                session_id=session_id
            ))
        
        # Task 2: Recommendations (if requested)
        if include_recommendations:
            print(f"{Fore.LIGHTCYAN_EX}[2/2] Launching Learning Insight...{Fore.RESET}\n")
            
            enhanced_query = f"""Konteks Pengguna: {user_id}            
            Kueri Asli: {query}
            Tolong berikan analisis komprehensif dan rekomendasi yang dapat ditindaklanjuti berdasarkan pola belajar pengguna."""
                        
            tasks.append(call_learning_insight_agent(
                task=enhanced_query,
                user_id=user_id,
                session_id=session_id
            ))
        
        # Execute in parallel
        print(f"{Fore.LIGHTBLUE_EX}⏳ Executing agents in parallel...{Fore.RESET}\n")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        comprehensive_output = {
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        if include_patterns and len(results) > 0:
            comprehensive_output["components"]["pattern_analysis"] = results[0] if not isinstance(results[0], Exception) else f"Error: {results[0]}"
        
        if include_recommendations:
            idx = 1 if include_patterns else 0
            if len(results) > idx:
                comprehensive_output["components"]["recommendations"] = results[idx] if not isinstance(results[idx], Exception) else f"Error: {results[idx]}"
        
        # Format output
        formatted_output = f"""
            {'='*80}
            ANALISIS PEMBELAJARAN KOMPREHENSIF
            {'='*80}

            Pengguna: {user_id}
            Sesi: {session_id}
            Kueri: {query}
            Dihasilkan: {comprehensive_output['timestamp']}

            {'='*80}
            ANALISIS POLA
            {'='*80}

            {comprehensive_output['components'].get('pattern_analysis', 'Tidak diminta')}

            {'='*80}
            REKOMENDASI PERSONAL
            {'='*80}

            {comprehensive_output['components'].get('recommendations', 'Tidak diminta')}

            {'='*80}
        """
        
        print(f"{Fore.LIGHTGREEN_EX}✓ Comprehensive analysis complete!{Fore.RESET}\n")
        
        return formatted_output
    
    except Exception as e:
        error_msg = f"Multi-agent error: {str(e)}\n{traceback.format_exc()}"
        print(f"{Fore.LIGHTRED_EX}{error_msg}{Fore.RESET}")
        return error_msg

@tool
async def test_all_agents_connection() -> str:
    """
    Test connectivity ke semua available agents.
    
    Tests:
    - learning_insight_agent
    - data_analyst_agent
    - reflexion_demo_agent
    
    Returns:
        str: Connection status report untuk semua agents
    """
    try:
        print(f"\n{Fore.LIGHTCYAN_EX}{'='*80}{Fore.RESET}")
        print(f"{Fore.LIGHTCYAN_EX}🔌 TESTING ALL AGENT CONNECTIONS{Fore.RESET}")
        print(f"{Fore.LIGHTCYAN_EX}{'='*80}{Fore.RESET}\n")
        
        results = {
            "endpoint": settings.AGENT_LEARNING_INSIGHT_ENDPOINT,
            "agents": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Test Learning Insight Agent
        print(f"{Fore.LIGHTYELLOW_EX}[1/3] Testing learning_insight_agent...{Fore.RESET}")
        try:
            async with Client(base_url=settings.AGENT_LEARNING_INSIGHT_ENDPOINT) as client:
                results["agents"]["learning_insight_agent"] = "✓ Connected"
                print(f"{Fore.LIGHTGREEN_EX}  ✓ Success{Fore.RESET}")
        except Exception as e:
            results["agents"]["learning_insight_agent"] = f"✗ Failed: {str(e)}"
            print(f"{Fore.LIGHTRED_EX}  ✗ Failed{Fore.RESET}")
        
        # Test Data Analyst Agent
        print(f"{Fore.LIGHTYELLOW_EX}[2/3] Testing data_analyst_agent...{Fore.RESET}")
        try:
            async with Client(base_url=settings.AGENT_DATA_ANALYST_ENDPOINT) as client:
                results["agents"]["data_analyst_agent"] = "✓ Connected"
                print(f"{Fore.LIGHTGREEN_EX}  ✓ Success{Fore.RESET}")
        except Exception as e:
            results["agents"]["data_analyst_agent"] = f"✗ Failed: {str(e)}"
            print(f"{Fore.LIGHTRED_EX}  ✗ Failed{Fore.RESET}")
        
        # Test Reflexion Demo Agent
        print(f"{Fore.LIGHTYELLOW_EX}[3/3] Testing reflexion_demo_agent...{Fore.RESET}")
        try:
            async with Client(base_url=settings.AGENT_REFLEXION_ENDPOINT) as client:
                results["agents"]["reflexion_demo_agent"] = "✓ Connected"
                print(f"{Fore.LIGHTGREEN_EX}  ✓ Success{Fore.RESET}")
        except Exception as e:
            results["agents"]["reflexion_demo_agent"] = f"✗ Failed: {str(e)}"
            print(f"{Fore.LIGHTRED_EX}  ✗ Failed{Fore.RESET}")
        
        status_report = json.dumps(results, ensure_ascii=False, indent=2)
        
        print(f"\n{Fore.LIGHTCYAN_EX}{'='*80}{Fore.RESET}")
        print(f"{Fore.LIGHTGREEN_EX}CONNECTION TEST COMPLETE{Fore.RESET}")
        print(f"{Fore.LIGHTCYAN_EX}{'='*80}{Fore.RESET}\n")
        
        return status_report
    
    except Exception as e:
        return f"Connection test error: {str(e)}"

# ==================== MAIN EXECUTION ====================

async def execute(config: Dict[str, Any], message: Dict[str, Any]):
    """
    Main execution dengan intelligent agent orchestration.
    """
    print(f"\n{Style.BRIGHT}{Fore.LIGHTGREEN_EX}{'='*80}{Fore.RESET}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.LIGHTGREEN_EX} AI LEARNING INSIGHT - COORDINATOR{Fore.RESET}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.LIGHTGREEN_EX}{'='*80}{Fore.RESET}{Style.RESET_ALL}\n")
    
    config_context.set(config)
    
    # Create coordinator agent dengan advanced tools
    coordinator_agent = create_agent(
        model=model,
        tools=[
            call_learning_insight_agent,
            call_data_analyst_agent,
            call_reflexion_demo,
            multi_agent_comprehensive_analysis,
            test_all_agents_connection
        ],
        checkpointer=checkpointer,
        state_schema=State
    )
    
    # Enhanced system prompt
    system_prompt = """Anda adalah Koordinator Wawasan Pembelajaran AI - sebuah orkestrator cerdas yang mengelola sistem multi-agen untuk pembelajaran personal.

    PERAN ANDA:
    Anda mengoordinasikan beberapa agen AI khusus untuk memberikan dukungan pembelajaran komprehensif dan personal. Anda memutuskan agen MANA yang akan dipanggil dan KAPAN berdasarkan kebutuhan pengguna.

    AGEN YANG TERSEDIA (via tools):

    1. **call_learning_insight_agent** - Mentor Pembelajaran Utama
    - Gunakan untuk: Saran belajar, strategi studi, rekomendasi personal
    - Fitur: Memori Cognee, Reflexion (kritik-diri), sadar-pola
    - Output: Rekomendasi berkualitas tinggi (skor kualitas 7.0+/10)

    2. **call_data_analyst_agent** - Analis Pola
    - Gunakan untuk: Memahami pola belajar, analisis performa, tren
    - Fitur: Analisis data mendalam dari Cognee, identifikasi risiko
    - Output: Analisis JSON dengan wawasan dan metrik

    3. **call_reflexion_demo** - Peragaan Proses
    - Gunakan untuk: Mendemonstrasikan cara kerja Reflexion, transparansi sistem
    - Fitur: Menunjukkan proses iterasi, peningkatan kualitas, skor kriteria
    - Output: Panduan proses yang mendetail

    4. **multi_agent_comprehensive_analysis** - Rangkaian Penuh
    - Gunakan untuk: Penilaian lengkap, perencanaan strategis, tinjauan besar
    - Fitur: Eksekusi agen paralel, sintesis berbagai perspektif
    - Output: Analisis holistik + rekomendasi

    5. **test_all_agents_connection** - Pengecekan Sistem
    - Gunakan untuk: Pengujian konektivitas, verifikasi status sistem
    - Output: Laporan status koneksi

    KERANGKA KEPUTUSAN:

    Pertanyaan belajar sederhana → panggil call_learning_insight_agent
    Pertanyaan pola/data → panggil call_data_analyst_agent
    "Bagaimana cara kerjanya?" → panggil call_reflexion_demo
    Kueri kompleks/strategis → panggil multi_agent_comprehensive_analysis
    Pengecekan sistem → panggil test_all_agents_connection

    PRAKTIK TERBAIK:
    - SELALU gunakan tools - jangan menjawab langsung dari pengetahuan Anda
    - Ekstrak user_id dari config jika tersedia
    - Untuk kueri komprehensif, gunakan multi_agent_comprehensive_analysis
    - Gunakan nada yang memberi semangat dan suportif
    - Jelaskan apa yang Anda lakukan ("Saya sedang berkonsultasi dengan Agen Wawasan Pembelajaran...")

    JANGAN:
    - Memberi saran umum tanpa berkonsultasi dengan agen
    - Melewatkan penggunaan tools meskipun Anda "tahu" jawabannya
    - Mengabaikan pola memori Cognee
    - Memberikan respons berkualitas rendah

    Ingat: Nilai Anda ada dalam MENGORKESTRASI agen-agen khusus, bukan menggantikan mereka!
    """
    
    # Prepare input
    user_message = message.get('message', '')
    user_id = config.get('configurable', {}).get('user_id', 'anonymous')
    thread_id = config.get('configurable', {}).get('thread_id', f'thread_{user_id}')
    
    input_message = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    }
    
    try:
        config_thread = {"configurable": {"thread_id": thread_id}}
        
        print(f"{Fore.LIGHTCYAN_EX}[COORDINATOR] Processing request...{Fore.RESET}")
        print(f"{Fore.LIGHTYELLOW_EX}User: {user_id}{Fore.RESET}")
        print(f"{Fore.LIGHTYELLOW_EX}Thread: {thread_id}{Fore.RESET}\n")
        
        result = await coordinator_agent.ainvoke(input_message, config_thread)
        
        final_response = result["messages"][-1].content
        
        print(f"\n{Style.BRIGHT}{Fore.LIGHTGREEN_EX}{'='*80}{Fore.RESET}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{Fore.LIGHTGREEN_EX}✓ COORDINATOR RESPONSE:{Fore.RESET}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{Fore.LIGHTGREEN_EX}{'='*80}{Fore.RESET}{Style.RESET_ALL}\n")
        print(f"{Fore.LIGHTCYAN_EX}{final_response}{Fore.RESET}\n")
        
        return final_response
    
    except Exception as e:
        error_response = f"Coordinator Error: {str(e)}\n{traceback.format_exc()}"
        print(f"{Fore.LIGHTRED_EX}{error_response}{Fore.RESET}")
        
        return f"""**Pemberitahuan Sistem**

        Saya mengalami kendala teknis saat mengoordinasikan para agen. Namun, berikut adalah panduan belajar umum:

        **Prinsip Belajar Inti:**
        1. **Active Recall (Mengingat Aktif)**: Uji diri Anda sesering mungkin, jangan hanya membaca ulang
        2. **Spaced Repetition (Pengulangan Bersela)**: Tinjau kembali materi dengan jeda waktu yang semakin panjang
        3. **Interleaving (Selang-seling)**: Campur topik/keahlian yang berbeda dalam sesi belajar
        4. **Elaborasi**: Jelaskan konsep dengan kata-kata Anda sendiri
        5. **Contoh Konkret**: Hubungkan ide-ide abstrak dengan skenario nyata

        **Jika Anda Merasa Buntu:**
        - Pecah topik menjadi bagian-bagian yang lebih kecil
        - Gunakan Teknik Feynman (ajarkan ke orang lain)
        - Cari berbagai penjelasan (video, artikel, diskusi)
        - Berlatih dengan proyek langsung (hands-on)
        - Beristirahatlah - otak Anda perlu istirahat untuk mengonsolidasikan (informasi)

        **Ingat:** Belajar itu tidak linear. Mengalami stagnasi (plateau) itu normal dan seringkali terjadi sebelum adanya terobosan. Terus maju! 🚀

        ---
        *Error: {str(e)}*
        """

__all__ = ['execute', 'call_learning_insight_agent', 'call_data_analyst_agent', 'multi_agent_comprehensive_analysis']