# MULTI-AGENT ORCHESTRATOR
from typing import List, Dict, Any, Optional
from datetime import datetime
from utils.logger import logger
from langchain_core.messages import HumanMessage

import asyncio
import json

from utils.assistant.agent_server.helper.reflexion_agent_config import ReflexionAgent

class MultiAgentOrchestrator:
    """
    Advanced Multi-Agent Orchestrator dengan:
    1. Intelligent agent routing
    2. Parallel agent execution (when possible)
    3. Result fusion and synthesis
    4. Cognee-powered coordination
    """
    
    def __init__(self, model, cognee_tools: Dict):
        self.model = model
        self.cognee_add = cognee_tools['add']
        self.cognee_search = cognee_tools['search']
        self.reflexion_agent = ReflexionAgent(
            model=model,
            cognee_add_tool=self.cognee_add,
            cognee_search_tool=self.cognee_search
        )
    
    async def process_learning_request(
        self,
        user_id: str,
        query: str,
        session_id: str,
        config: Dict
    ) -> Dict[str, Any]:
        """
        Main orchestration workflow dengan intelligent routing
        """
        logger.info(f"[ORCHESTRATOR] Processing request for user {user_id}, session {session_id}")
        
        # Step 1: Query classification & intent detection
        intent = await self._classify_query_intent(query)
        logger.info(f"[ORCHESTRATOR] Detected intent: {intent['type']}")
        
        # Step 2: Query Cognee untuk comprehensive context
        user_context = await self._build_comprehensive_context(
            user_id=user_id,
            query=query,
            session_id=session_id
        )
        
        # Step 3: Route to appropriate workflow
        if intent['type'] == 'pattern_analysis':
            result = await self._handle_pattern_analysis(user_id, query, session_id, user_context)
        elif intent['type'] == 'learning_recommendation':
            result = await self._handle_learning_recommendation(user_id, query, session_id, user_context)
        elif intent['type'] == 'motivation':
            result = await self._handle_motivation_request(user_id, query, session_id, user_context)
        elif intent['type'] == 'comprehensive':
            result = await self._handle_comprehensive_analysis(user_id, query, session_id, user_context)
        else:
            # Default: learning recommendation
            result = await self._handle_learning_recommendation(user_id, query, session_id, user_context)
        
        # Step 4: Store interaction ke Cognee
        await self._store_interaction_to_cognee(
            user_id=user_id,
            session_id=session_id,
            query=query,
            result=result,
            intent=intent
        )
        
        return result
    
    async def _classify_query_intent(self, query: str) -> Dict[str, Any]:
        """Classify user query intent using LLM"""
        
        classification_prompt = f"""Klasifikasikan kueri pengguna berikut ke dalam salah satu niat ini:

        1. pattern_analysis: Pengguna ingin memahami pola belajar, riwayat, analisis keberhasilan/kegagalan mereka.
        2. learning_recommendation: Pengguna membutuhkan saran belajar, cara meningkatkan diri, strategi belajar.
        3. motivation: Pengguna membutuhkan dorongan semangat, merasa buntu, kehilangan motivasi.
        4. comprehensive: Pengguna ingin analisis penuh + rekomendasi + pola.

        Kueri: "{query}"

        Kembalikan JSON: {{"type": "tipe_niat", "confidence": 0.0-1.0, "keywords": ["kata_kunci1", "kata_kunci2"]}}"""
        
        try:
            result = await self.model.ainvoke([HumanMessage(content=classification_prompt)])
            intent_data = json.loads(result.content)
            return intent_data
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] Intent classification failed: {e}")
            return {"type": "learning_recommendation", "confidence": 0.5, "keywords": []}
    
    async def _build_comprehensive_context(
        self,
        user_id: str,
        query: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Build comprehensive context dari Cognee memory"""
        
        # Parallel search untuk different aspects
        tasks = [
            self._search_cognee(f"learning history and past interactions for user {user_id}"),
            self._search_cognee(f"success patterns and achievements for user {user_id}"),
            self._search_cognee(f"failure patterns and struggles for user {user_id}"),
            self._search_cognee(f"learning preferences and style for user {user_id}")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        context = {
            "user_id": user_id,
            "session_id": session_id,
            "learning_history": results[0] if not isinstance(results[0], Exception) else "",
            "success_patterns": results[1] if not isinstance(results[1], Exception) else "",
            "failure_patterns": results[2] if not isinstance(results[2], Exception) else "",
            "learning_preferences": results[3] if not isinstance(results[3], Exception) else "",
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"[ORCHESTRATOR] Built comprehensive context with {sum(1 for r in results if not isinstance(r, Exception))} successful searches")
        
        return context
    
    async def _search_cognee(self, query: str) -> str:
        """Helper untuk search Cognee"""
        try:
            result = await self.cognee_search.ainvoke({"query": query})
            return result if result else ""
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] Cognee search failed for '{query}': {e}")
            return ""
    
    async def _handle_pattern_analysis(
        self,
        user_id: str,
        query: str,
        session_id: str,
        context: Dict
    ) -> Dict[str, Any]:
        """Handle pattern analysis requests"""
        logger.info(f"[ORCHESTRATOR] Executing pattern analysis workflow")
        
        # Aggregate patterns dari context
        analysis = {
            "user_id": user_id,
            "analysis_type": "pattern_analysis",
            "patterns": {
                "success_patterns": context.get('success_patterns', ''),
                "failure_patterns": context.get('failure_patterns', ''),
                "learning_preferences": context.get('learning_preferences', '')
            },
            "insights": await self._generate_pattern_insights(context),
            "recommendations": "Based on patterns, consider focusing on areas where you've shown success before.",
            "timestamp": datetime.now().isoformat()
        }
        
        return analysis
    
    async def _handle_learning_recommendation(
        self,
        user_id: str,
        query: str,
        session_id: str,
        context: Dict
    ) -> Dict[str, Any]:
        """Handle learning recommendation with Reflexion"""
        logger.info(f"[ORCHESTRATOR] Executing learning recommendation workflow with Reflexion")
        
        # Use Reflexion Agent untuk generate high-quality recommendation
        reflexion_result = await self.reflexion_agent.generate_with_reflection(
            query=query,
            user_context=context,
            session_id=session_id
        )
        
        return {
            "user_id": user_id,
            "response_type": "learning_recommendation",
            "recommendation": reflexion_result['final_response'],
            "quality_score": reflexion_result['final_score'],
            "reflexion_iterations": reflexion_result['iterations'],
            "improvement": reflexion_result['improvement'],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_motivation_request(
        self,
        user_id: str,
        query: str,
        session_id: str,
        context: Dict
    ) -> Dict[str, Any]:
        """Handle motivation requests dengan empathy"""
        logger.info(f"[ORCHESTRATOR] Executing motivation workflow")
        
        # Generate motivational response dengan context dari successes
        motivation_prompt = f"""Konteks Pengguna:
        {json.dumps(context, ensure_ascii=False)}

        Pengguna membutuhkan motivasi dan dorongan semangat. Mereka berkata: "{query}"

        Berikan respons motivasi yang memberdayakan, hangat, dan tulus yang:
        1. Mengakui perasaan mereka
        2. Menyoroti kesuksesan mereka di masa lalu (dari konteks)
        3. Membingkai ulang tantangan sebagai peluang bertumbuh
        4. Memberikan langkah-langkah kecil spesifik berikutnya
        5. Diakhiri dengan dorongan semangat yang tulus

        Respons:"""
        
        result = await self.model.ainvoke([HumanMessage(content=motivation_prompt)])
        
        return {
            "user_id": user_id,
            "response_type": "motivation",
            "message": result.content,
            "success_reminders": context.get('success_patterns', ''),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_comprehensive_analysis(
        self,
        user_id: str,
        query: str,
        session_id: str,
        context: Dict
    ) -> Dict[str, Any]:
        """Handle comprehensive analysis - kombinasi pattern + recommendation"""
        logger.info(f"[ORCHESTRATOR] Executing comprehensive analysis workflow")
        
        # Execute both workflows in parallel
        pattern_task = self._handle_pattern_analysis(user_id, query, session_id, context)
        recommendation_task = self._handle_learning_recommendation(user_id, query, session_id, context)
        
        pattern_result, recommendation_result = await asyncio.gather(
            pattern_task,
            recommendation_task
        )
        
        # Synthesize results
        comprehensive_result = {
            "user_id": user_id,
            "response_type": "comprehensive",
            "pattern_analysis": pattern_result,
            "personalized_recommendation": recommendation_result,
            "synthesized_insight": await self._synthesize_insights(pattern_result, recommendation_result),
            "timestamp": datetime.now().isoformat()
        }
        
        return comprehensive_result
    
    async def _generate_pattern_insights(self, context: Dict) -> str:
        """Generate insights dari patterns"""
        prompt = f"""Analisis pola pembelajaran ini dan berikan wawasan yang dapat ditindaklanjuti:

        Pola Keberhasilan:
        {context.get('success_patterns', 'Tidak ada data')}

        Pola Kegagalan:
        {context.get('failure_patterns', 'Tidak ada data')}

        Preferensi Belajar:
        {context.get('learning_preferences', 'Tidak ada data')}

        Berikan 3-5 wawasan utama:"""
        
        result = await self.model.ainvoke([HumanMessage(content=prompt)])
        return result.content
    
    async def _synthesize_insights(
        self,
        pattern_result: Dict,
        recommendation_result: Dict
    ) -> str:
        """Synthesize insights dari multiple analyses"""
        
        synthesis_prompt = f"""Sintesis (gabungkan) kedua analisis ini menjadi satu wawasan yang kohesif:

        Analisis Pola:
        {json.dumps(pattern_result, ensure_ascii=False, indent=2)}

        Rekomendasi:
        {json.dumps(recommendation_result, ensure_ascii=False, indent=2)}

        Berikan wawasan sintesis yang menggabungkan keduanya (2-3 paragraf):"""
        
        result = await self.model.ainvoke([HumanMessage(content=synthesis_prompt)])
        return result.content
    
    async def _store_interaction_to_cognee(
        self,
        user_id: str,
        session_id: str,
        query: str,
        result: Dict,
        intent: Dict
    ):
        """Store interaction ke Cognee untuk future reference"""
        try:
            interaction_summary = f"""
                Ringkasan Interaksi Pembelajaran

                Pengguna: {user_id}
                Sesi: {session_id}
                Niat: {intent['type']}
                Waktu: {datetime.now().isoformat()}

                Pertanyaan: {query}

                Jenis Hasil: {result.get('response_type', 'tidak diketahui')}
                Skor Kualitas: {result.get('quality_score', 'N/A')}

                Interaksi ini memberikan konteks untuk memahami perjalanan pembelajaran pengguna.
            """

            await self.cognee_add.ainvoke({"data": interaction_summary})
            logger.info(f"[ORCHESTRATOR] ✓ Stored interaction to Cognee")
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] Could not store interaction: {e}")