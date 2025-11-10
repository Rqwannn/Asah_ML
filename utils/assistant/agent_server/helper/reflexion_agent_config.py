# REFLEXION AGENT

from app.config import settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from utils.logger import logger
from typing import List, Dict, Any
from datetime import datetime
import json

class ReflexionAgent:
    """
    Advanced Reflexion Agent dengan:
    1. Multi-criteria evaluation
    2. Adaptive iteration strategy
    3. Learning from past critiques (via Cognee)
    4. Explainable critique reasoning
    """
    
    def __init__(self, model, cognee_add_tool, cognee_search_tool):
        self.model = model
        self.cognee_add = cognee_add_tool
        self.cognee_search = cognee_search_tool
        self.max_iterations = settings.REFLEXION_MAX_ITERATIONS
        self.min_quality_score = settings.REFLEXION_MIN_QUALITY_SCORE
        
        # multi-criteria evaluation
        self.critique_prompt = ChatPromptTemplate.from_messages([
            ("system", """Anda adalah Kritikus Pembelajaran AI Canggih yang mengevaluasi rekomendasi pembelajaran dengan kerangka kerja multi-kriteria.

            `KRITERIA EVALUASI (Beri skor masing-masing 0-10):

            1. **Personalisasi** (Bobot 30%)
            - Seberapa personal rekomendasi ini dengan riwayat belajar pengguna?
            - Apakah mempertimbangkan gaya belajar individu?
            - Apakah memanfaatkan pola kesuksesan dari memori Cognee?

            2. **Kelayakan Tindak Lanjut (Actionability)** (Bobot 25%)
            - Apakah rekomendasinya konkret dan dapat dilaksanakan?
            - Apakah ada linimasa dan tonggak pencapaian (milestone) yang jelas?
            - Apakah mudah diimplementasikan oleh pengguna?

            3. **Kebenaran Pedagogis** (Bobot 20%)
            - Apakah menggunakan teori pembelajaran yang valid?
            - Apakah tingkat kesulitannya progresif (bertahap) dengan tepat?
            - Apakah menggunakan metode berbasis bukti (evidence-based)?

            4. **Dampak Motivasional** (Bobot 15%)
            - Apakah menginspirasi dan memberi semangat?
            - Apakah mengatasi hambatan emosional?
            - Apakah membangun kepercayaan diri?

            5. **Relevansi Kontekstual** (Bobot 10%)
            - Apakah sesuai dengan situasi pengguna saat ini?
            - Apakah mempertimbangkan batasan waktu/sumber daya?
            - Apakah selaras dengan tujuan pengguna?

            Kembalikan dalam format JSON:
            {{
                "scores": {{
                    "personalization": 8.5,
                    "actionability": 7.0,
                    "pedagogical_soundness": 8.0,
                    "motivational_impact": 6.5,
                    "contextual_relevance": 9.0
                }},
                "weighted_score": 7.8,
                "strengths": ["kekuatan 1", "kekuatan 2", "kekuatan 3"],
                "weaknesses": ["kelemahan 1", "kelemahan 2"],
                "critical_issues": ["isu kritis 1 jika ada"],
                "refinement_strategy": "strategi spesifik untuk perbaikan",
                "reasoning": "penjelasan mendetail mengapa skor ini diberikan"
            }}"""),
                        ("user", """Kueri Asli: {query}

            Respons yang Dihasilkan: {response}

            Konteks Pengguna dari Cognee:
            {context}

            Pola Riwayat Pembelajaran:
            {history_patterns}

            Berikan evaluasi multi-kriteria yang mendetail:""")
        ])
        
        self.refine_prompt = ChatPromptTemplate.from_messages([
        ("system", """Anda adalah Penyempurna Respons Pembelajaran Adaptif yang ahli dalam psikologi pendidikan dan pembelajaran personal.
            PRIORITAS PENYEMPURNAAN (berdasarkan kritik):
            1. Perbaiki ISU KRITIS terlebih dahulu (jika ada)
            2. Perkuat KELEMAHAN (fokus pada skor terendah)
            3. Tingkatkan KEKUATAN (buat menjadi lebih baik lagi)
            4. Terapkan STRATEGI PENYEMPURNAAN

            PRINSIP PENYEMPURNAAN:
            - Pertahankan gaya bahasa dan konteks pengguna
            - Tambahkan kekhususan dan hal-hal konkret
            - Sertakan langkah-langkah mikro untuk 'kemenangan cepat' (quick wins)
            - Seimbangkan tantangan dengan ketercapaian
            - Masukkan elemen motivasi secara alami

            Output harus:
            ✓ Lebih personal (gunakan pola historis)
            ✓ Lebih dapat ditindaklanjuti (langkah spesifik + linimasa)
            ✓ Lebih menarik (elemen penceritaan/storytelling)
            ✓ Berbasis bukti (prinsip ilmu pembelajaran)"""),
                        ("user", """Respons Asli:
            {response}

            Analisis Kritik:
            {critique}

            Konteks Pengguna:
            {context}

            Pola Kesuksesan dari Memori Cognee:
            {success_patterns}

            Buat respons yang TELAH DISEMPURNAKAN yang mengatasi semua poin kritik:""")
        ])
    
    async def generate_with_reflection(
        self, 
        query: str, 
        user_context: Dict,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Advanced reflexion loop dengan learning from past critiques
        """
        logger.info(f"[REFLEXION] Starting advanced reflexion for session {session_id}")
        
        # Step 1: Search Cognee untuk learning patterns user
        history_patterns = await self._get_user_learning_patterns(
            user_id=user_context.get('user_id', 'unknown'),
            session_id=session_id
        )
        
        # Step 2: Search past successful critiques untuk learn
        past_critiques = await self._get_past_successful_critiques(session_id)
        
        # Step 3: Generate initial response dengan context
        initial_response = await self._generate_initial_response(
            query=query,
            context=user_context,
            patterns=history_patterns,
            past_learnings=past_critiques
        )
        
        best_response = initial_response
        best_score = 0
        critique_history = []
        
        # Step 4: Reflexion loop dengan adaptive strategy
        for iteration in range(self.max_iterations):
            logger.info(f"[REFLEXION] Iteration {iteration + 1}/{self.max_iterations}")
            
            # Critique dengan multi-criteria
            critique_result = await self._multi_criteria_critique(
                query=query,
                response=best_response,
                context=user_context,
                history_patterns=history_patterns
            )
            
            current_score = critique_result['weighted_score']
            critique_history.append({
                'iteration': iteration + 1,
                'score': current_score,
                'critique': critique_result
            })
            
            logger.info(f"[REFLEXION] Iteration {iteration + 1} Score: {current_score:.2f}/10")
            logger.info(f"[REFLEXION] Breakdown: {critique_result['scores']}")
            
            # Early stopping jika sudah excellent
            if current_score >= self.min_quality_score:
                logger.info(f"[REFLEXION] ✓ Achieved target score ({current_score:.2f} >= {self.min_quality_score})")
                
                # Store successful critique pattern ke Cognee
                await self._store_successful_critique(
                    session_id=session_id,
                    critique=critique_result,
                    query=query,
                    response=best_response
                )
                break
            
            # Check jika ada critical issues
            if critique_result.get('critical_issues'):
                logger.warning(f"[REFLEXION] Critical issues found: {critique_result['critical_issues']}")
            
            # Refine response
            success_patterns = await self._extract_success_patterns_from_cognee(
                user_id=user_context.get('user_id'),
                session_id=session_id
            )
            
            refined_response = await self._refine_response(
                response=best_response,
                critique=critique_result,
                context=user_context,
                success_patterns=success_patterns
            )
            
            # Update best response jika improve
            if current_score > best_score:
                best_response = refined_response
                best_score = current_score
                logger.info(f"[REFLEXION] ✓ Improvement: {best_score:.2f} (↑ from previous)")
            else:
                logger.info(f"[REFLEXION] No improvement, keeping previous best")
                # Stop jika tidak improve (diminishing returns)
                break
        
        # Step 5: Store final reflection ke Cognee
        final_reflection = {
            "query": query,
            "final_response": best_response,
            "final_score": best_score,
            "iterations": len(critique_history),
            "critique_history": critique_history,
            "user_id": user_context.get('user_id'),
            "timestamp": datetime.now().isoformat(),
            "outcome": "excellent" if best_score >= 8.5 else "good" if best_score >= 7.0 else "acceptable"
        }
        
        await self._store_reflection_to_cognee(
            reflection=final_reflection,
            session_id=session_id
        )
        
        return {
            "final_response": best_response,
            "final_score": best_score,
            "iterations": len(critique_history),
            "critique_history": critique_history,
            "improvement": best_score - critique_history[0]['score'] if critique_history else 0
        }
    
    async def _get_user_learning_patterns(self, user_id: str, session_id: str) -> Dict:
        """Query Cognee untuk learning patterns user"""
        try:
            search_query = f"learning patterns and history for user {user_id}, including success and failure patterns"
            results = await self.cognee_search.ainvoke({"query": search_query})
            
            return {
                "raw_results": results,
                "has_history": bool(results)
            }
        except Exception as e:
            logger.warning(f"[REFLEXION] Could not fetch learning patterns: {e}")
            return {"raw_results": "", "has_history": False}
    
    async def _get_past_successful_critiques(self, session_id: str) -> List[Dict]:
        """Learn from past successful critiques"""
        try:
            search_query = "successful critique patterns with high scores above 8.0"
            results = await self.cognee_search.ainvoke({"query": search_query})
            return results if results else []
        except Exception as e:
            logger.warning(f"[REFLEXION] Could not fetch past critiques: {e}")
            return []
    
    async def _generate_initial_response(
        self, 
        query: str, 
        context: Dict,
        patterns: Dict,
        past_learnings: List
    ) -> str:
        """Generate initial response dengan rich context"""
        
        context_str = json.dumps(context, ensure_ascii=False, indent=2)
        patterns_str = json.dumps(patterns, ensure_ascii=False, indent=2)
        learnings_str = json.dumps(past_learnings[:3], ensure_ascii=False, indent=2) if past_learnings else "No past learnings"
        
        prompt = f"""Anda adalah Mentor Pembelajaran AI yang ahli dalam pendidikan personal.

        KONTEKS PENGGUNA:
        {context_str}

        POLA PEMBELAJARAN (dari Memori Cognee):
        {patterns_str}

        PENDEKATAN SUKSES SEBELUMNYA:
        {learnings_str}

        KUERI PENGGUNA:
        {query}

        Hasilkan rekomendasi pembelajaran personal yang:
        1. Sangat personal berdasarkan pola
        2. Dapat ditindaklanjuti dengan langkah-langkah spesifik
        3. Memotivasi dan memberdayakan
        4. Pendekatan berbasis bukti
        5. Menyertakan linimasa dan tonggak pencapaian (milestones)

        Respons:"""
        
        result = await self.model.ainvoke([HumanMessage(content=prompt)])
        return result.content
    
    async def _multi_criteria_critique(
        self,
        query: str,
        response: str,
        context: Dict,
        history_patterns: Dict
    ) -> Dict:
        """Multi-criteria evaluation"""
        
        messages = self.critique_prompt.format_messages(
            query=query,
            response=response,
            context=json.dumps(context, ensure_ascii=False),
            history_patterns=json.dumps(history_patterns, ensure_ascii=False)
        )
        
        result = await self.model.ainvoke(messages)
        
        # Parse JSON response
        try:
            critique_data = json.loads(result.content)
            return critique_data
        except json.JSONDecodeError:
            logger.error(f"[REFLEXION] Failed to parse critique JSON: {result.content}")
            # Fallback simple critique
            return {
                "scores": {
                    "personalization": 6.0,
                    "actionability": 6.0,
                    "pedagogical_soundness": 6.0,
                    "motivational_impact": 6.0,
                    "contextual_relevance": 6.0
                },
                "weighted_score": 6.0,
                "strengths": ["Response provided"],
                "weaknesses": ["Needs improvement"],
                "critical_issues": [],
                "refinement_strategy": "Add more personalization and specific steps",
                "reasoning": "Fallback critique due to parsing error"
            }
    
    async def _extract_success_patterns_from_cognee(
        self,
        user_id: str,
        session_id: str
    ) -> Dict:
        """Extract success patterns dari Cognee"""
        try:
            query = f"successful learning approaches and patterns for user {user_id}"
            results = await self.cognee_search.ainvoke({"query": query})
            return {"patterns": results} if results else {"patterns": "No patterns found"}
        except Exception as e:
            logger.warning(f"[REFLEXION] Could not extract success patterns: {e}")
            return {"patterns": "Error fetching patterns"}
    
    async def _refine_response(
        self,
        response: str,
        critique: Dict,
        context: Dict,
        success_patterns: Dict
    ) -> str:
        """Refine response berdasarkan critique"""
        
        messages = self.refine_prompt.format_messages(
            response=response,
            critique=json.dumps(critique, ensure_ascii=False, indent=2),
            context=json.dumps(context, ensure_ascii=False),
            success_patterns=json.dumps(success_patterns, ensure_ascii=False)
        )
        
        result = await self.model.ainvoke(messages)
        return result.content
    
    async def _store_successful_critique(
        self,
        session_id: str,
        critique: Dict,
        query: str,
        response: str
    ):
        """Store successful critique pattern ke Cognee untuk future learning"""
        try:
            critique_summary = f"""
                Pola Kritik Sukses (Skor: {critique['weighted_score']:.2f}/10)

                Kueri: {query}

                Kualitas Respons:
                - Personalisasi: {critique['scores']['personalization']:.1f}/10
                - Kelayakan Tindak Lanjut: {critique['scores']['actionability']:.1f}/10
                - Kebenaran Pedagogis: {critique['scores']['pedagogical_soundness']:.1f}/10
                - Dampak Motivasional: {critique['scores']['motivational_impact']:.1f}/10
                - Relevansi Kontekstual: {critique['scores']['contextual_relevance']:.1f}/10

                Kekuatan: {', '.join(critique['strengths'])}
                Strategi Penyempurnaan yang Digunakan: {critique['refinement_strategy']}

                Pola ini dapat digunakan kembali untuk kueri serupa.
            """
            await self.cognee_add.ainvoke({"data": critique_summary})
            logger.info(f"[REFLEXION] ✓ Stored successful critique pattern to Cognee")
        except Exception as e:
            logger.warning(f"[REFLEXION] Could not store critique: {e}")
    
    async def _store_reflection_to_cognee(self, reflection: Dict, session_id: str):
        """Store complete reflection ke Cognee"""
        try:
            reflection_text = f"""
                Refleksi Interaksi Pembelajaran

                Pengguna: {reflection['user_id']}
                Kueri: {reflection['query']}
                Skor Kualitas Akhir: {reflection['final_score']:.2f}/10
                Iterasi: {reflection['iterations']}
                Hasil: {reflection['outcome']}
                Waktu: {reflection['timestamp']}

                Respons Akhir:
                {reflection['final_response']}

                Lintasan Peningkatan:
                {json.dumps(reflection['critique_history'], ensure_ascii=False, indent=2)}
            """
            await self.cognee_add.ainvoke({"data": reflection_text})
            logger.info(f"[REFLEXION] ✓ Stored complete reflection to Cognee")
        except Exception as e:
            logger.warning(f"[REFLEXION] Could not store reflection: {e}")