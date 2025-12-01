"""
Database-only conversation engine - all responses grounded in retrieved activities
"""
from __future__ import annotations

import re
import json
import numpy as np
import traceback
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
import copy

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.ai_services import openai_service
from data.processor import data_processor
from data.time_processor import time_processor
from data.content_cache import content_cache
from services.content_retrieval import content_retrieval
from processing.document_processor import document_processor, parse_legacy_activity, strip_added_headers
from services.router import route_intent_and_target
from utils.selection import parse_selection_index
from services.conversation_manager import conversation_manager
from services.activity_customizer import activity_customizer
from utils.markdown_merger import merge_block_into_markdown
from helpers.session_sanitize import to_plain_activity, sanitize_session_data
from services.query_middleware import (
    encode_similar_constraint,
    neutralize_if_vague,
    merge_time_filters,
)


def _build_capabilities_payload():
    return {
        "type": "capabilities",
        "content": (
            "Hi there! 👋 I can help you:\n\n"
            "• **Find Activities** by time, topic, or grade\n"
            "• **Create Lesson Plans** with materials & directions\n"
            "• **Customize** activities (shorten, expand, adapt for different grades)\n"
            "• **Answer** teaching questions\n\n"
            "Try: *\"find 15 minute activities\"*, *\"show me affirmation activities\"*, "
            "or *\"open the first one\"*."
        ),
        "quick_actions": [
            {"label": "Find 15-min Activities", "message": "find 15 minute activities"},
            {"label": "Affirmation Activities", "message": "show me affirmation activities"},
            {"label": "Grade 5 Ideas", "message": "find activities for 5th grade"},
            {"label": "Build a Lesson Plan", "message": "create a lesson plan for goal setting (30 min)"},
        ],
        "show_shortlist": False,
        "show_download": False,
    }


class DBOnlyConversationEngine:
    """Database-only conversation engine - all responses are grounded in retrieved activities."""
    
    def __init__(self):
        # ====================================================================
        # AUTO-INITIALIZE DATA PROCESSOR (Best Solution)
        # ====================================================================
        if data_processor.df is None or data_processor.index is None:
            print("🔍 [ENGINE] Auto-initializing data processor...")
            import os
            
            # FIX: Get correct path relative to project root, not this file
            # Go up from: src/services/db_only_conversation_engine.py
            # To reach:   data/activities.csv
            
            current_file = os.path.abspath(__file__)
            # /Users/.../chatbot copy/src/services/db_only_conversation_engine.py
            
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            # /Users/.../chatbot copy
            
            csv_path = os.path.join(project_root, "data", "activities.csv")
            # /Users/.../chatbot copy/data/activities.csv
            
            print(f"🔍 [ENGINE] CSV path: {csv_path}")
            
            if not os.path.exists(csv_path):
                raise RuntimeError(f"CSV file not found at: {csv_path}")
            
            data_processor.initialize(csv_path)
            print("✅ [ENGINE] Data processor ready")
        
        # Now continue with rest of __init__
        self.conversation_history = []
        self.current_context = {}
        
        # Session tracking for multi-tier activity matching
        self.last_results = []
        self.session_history = []
        self.all_activities_seen = {}
        self.active_activity = None
        
        try:
            print("🔧 USE_LLM_RERANK =", os.getenv("USE_LLM_RERANK"))
        except Exception:
            pass
        
        self.specificity = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts: list[str] = []
            try:
                if hasattr(data_processor, 'df') and data_processor.df is not None:
                    n = len(data_processor.df) if data_processor.df is not None else 0
                    if n > 0:
                        for i in range(min(n, 2000)):
                            try:
                                row = data_processor.get_row(i)
                                title = data_processor.safe_get(row, "Strategic Action") or data_processor.safe_get(row, "Title") or ""
                                summary = data_processor.safe_get(row, "Short Description") or data_processor.safe_get(row, "Notes") or data_processor.safe_get(row, "Directions") or ""
                                texts.append(f"{title} {summary}")
                            except Exception:
                                continue
                else:
                    print("🔧 Specificity: data_processor not initialized yet, skipping")
            except Exception:
                texts = []

            if texts:
                import numpy as _np
                vec = TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words='english',
                    lowercase=True,
                    strip_accents='unicode',
                    min_df=1,
                )
                vec.fit(texts)
                idf = vec.idf_
                vocab = vec.vocabulary_
                idf_threshold = float(_np.percentile(idf, 35))
                ultra_common = {term for term, idx in vocab.items() if idf[idx] <= float(_np.percentile(idf, 10))}

                class _Specificity:
                    def __init__(self, vectorizer, idf, vocab, idf_threshold, ultra_common):
                        self.vectorizer = vectorizer
                        self.idf = idf
                        self.vocab = vocab
                        self.idf_threshold = idf_threshold
                        self.ultra_common = ultra_common

                    def max_idf_in_query(self, query: str):
                        try:
                            tokens = self.vectorizer.transform([query.lower()])
                            indices = tokens.indices
                            if len(indices) == 0:
                                return 0.0
                            max_idf = max(self.idf[idx] for idx in indices)
                            return float(max_idf)
                        except Exception:
                            return 0.0

                self.specificity = _Specificity(vec, idf, vocab, idf_threshold, ultra_common)
                print("✅ [ENGINE] Specificity analyzer ready")
        except Exception as e:
            print(f"⚠️  [ENGINE] Could not initialize specificity: {e}")

    def process_message(self, user_input: str, session_data: Dict) -> Dict[str, Any]:
        print(f"🔍 [DEBUG] Processing: {user_input[:100]}")

        try:
            conversation_history = session_data.get("messages", [])
            if not isinstance(conversation_history, list):
                conversation_history = []
            
            last_results = session_data.get("last_results", []) or self.last_results
            all_activities_seen = self.all_activities_seen
            active_focus = session_data.get("active_focus")
            
            # Route the message using router function
            from services.ai_services import openai_service
            routed = route_intent_and_target(
                user_input,
                conversation_history=conversation_history,
                last_results=last_results,
                active_focus=active_focus,
                session_data=session_data,  # Pass full session for context
                openai_service=openai_service
            )
            
            intent = routed.get("intent", "db_search")
            
            print(f"🔍 [DEBUG] db_only: Intent = {intent}")
            
            print(f"🎯 [DEBUG] Router returned:")
            print(f"   - intent: {routed.get('intent')}")
            print(f"   - confidence: {routed.get('confidence', 0):.2f}")
            print(f"   - source: {routed.get('source')}")
            print(f"   - target: {routed.get('target')}")
            
            # ============================================================
            # CRITICAL FIX: Check capabilities IMMEDIATELY after routing
            # This MUST come before ANY other intent processing
            # ============================================================
            if intent == "capabilities":
                print("✅ [DEBUG] Capabilities intent detected - returning capabilities payload")
                result = _build_capabilities_payload()
                if 'content' in result:
                    session_data['last_bot_message'] = result['content'][:500]
                return result

            confidence = routed.get("confidence", 0.0)
            source = routed.get("source", "unknown")
            target = routed.get("target")

            # Store metadata for downstream logging/testing
            intent_metadata = {
                "detected_intent": intent,
                "intent_confidence": confidence,
                "intent_source": source,
                "intent_target": target
            }

            # Handle error intent (invalid references)
            if intent == "error":
                error_message = routed.get("message", "I encountered an issue processing your request.")
                result = {
                    "type": "response",
                    "content": error_message,
                    "show_shortlist": len(last_results) > 0,
                    "activities": last_results if len(last_results) > 0 else [],
                    "metadata": {**intent_metadata, "error_type": routed.get("error_type")}
                }
                if 'content' in result:
                    session_data['last_bot_message'] = result['content'][:500]
                return result


            if intent == "teaching_question":
                result = self._handle_teaching_question(user_input, session_data)
                result["metadata"] = {**result.get("metadata", {}), **intent_metadata}
                if 'content' in result:
                    session_data['last_bot_message'] = result['content'][:500]
                return result

            if intent == "acknowledgment":
                result = self._handle_acknowledgment(user_input, routed, session_data)
                result["metadata"] = {**result.get("metadata", {}), **intent_metadata}
                if 'content' in result:
                    session_data['last_bot_message'] = result['content'][:500]
                return result

            if intent == "open_activity":
                result = self._handle_open_activity(user_input, routed, session_data)
                result["metadata"] = {**result.get("metadata", {}), **intent_metadata}
                if 'content' in result:
                    session_data['last_bot_message'] = result.get('content', result.get('chat_message', ''))[:500]
                return result

            if intent == "followup_question":
                active = session_data.get("active_focus")
                if not active:
                    print(f"⚠️  [DEBUG] followup_question but no active focus - treating as search")
                    result = self._handle_db_search(user_input, session_data)
                else:
                    result = self._handle_followup_question(user_input, session_data)
                result["metadata"] = {**result.get("metadata", {}), **intent_metadata}
                if 'content' in result:
                    session_data['last_bot_message'] = result['content'][:500]
                return result
            
            if intent == "lost_context":
                # User wants to modify but no activity is open
                clarification = self._generate_lost_context_clarification(user_input, session_data)
                result = {
                    "type": "response",
                    "content": clarification,
                    "show_shortlist": len(session_data.get("last_results", [])) > 0,
                    "activities": session_data.get("last_results", [])[:8] if session_data.get("last_results") else []
                }
                result["metadata"] = {**result.get("metadata", {}), **intent_metadata}
                if 'content' in result:
                    session_data['last_bot_message'] = result['content'][:500]
                return result
            
            if intent == "clarification_needed":
                # User referenced multiple activities - ask which one they want
                clarification_message = routed.get("clarification_message") or \
                    "I can only open one activity at a time. Which one would you like to open?"
                result = {
                    "type": "response",
                    "content": clarification_message,
                    "show_shortlist": len(session_data.get("last_results", [])) > 0,
                    "activities": session_data.get("last_results", [])[:8] if session_data.get("last_results") else []
                }
                result["metadata"] = {**result.get("metadata", {}), **intent_metadata}
                if 'content' in result:
                    session_data['last_bot_message'] = result['content'][:500]
                return result
            
            if intent == "multi_intent":
                # User has multiple intents in one message - execute them sequentially
                print(f"🔄 [MULTI_INTENT] Executing compound request")
                intent_sequence = routed.get("intent_sequence", [])
                
                if not intent_sequence:
                    print(f"⚠️  [MULTI_INTENT] No intent_sequence provided, falling back to db_search")
                    result = self._handle_db_search(user_input, session_data)
                    result["metadata"] = {**result.get("metadata", {}), **intent_metadata}
                    return result
                
                # Execute each intent in sequence
                combined_responses = []
                final_result = None
                
                for idx, intent_step in enumerate(intent_sequence, 1):
                    step_intent = intent_step.get("intent")
                    step_target = intent_step.get("target")
                    step_reasoning = intent_step.get("reasoning", "")
                    
                    print(f"🔄 [MULTI_INTENT] Step {idx}/{len(intent_sequence)}: {step_intent} - {step_reasoning}")
                    
                    # Create a synthetic routed object for this step
                    step_routed = {
                        "intent": step_intent,
                        "target": step_target,
                        "confidence": routed.get("confidence", 0.9),
                        "source": "multi_intent_sequence",
                        "reasoning": step_reasoning
                    }
                    
                    # Execute the intent
                    if step_intent == "db_search":
                        step_result = self._handle_db_search(user_input, session_data)
                    elif step_intent == "open_activity":
                        step_result = self._handle_open_activity(user_input, step_routed, session_data)
                    elif step_intent == "followup_customize":
                        step_result = self._handle_followup_customize(user_input, session_data)
                    elif step_intent == "followup_question":
                        step_result = self._handle_followup_question(user_input, session_data)
                    else:
                        print(f"⚠️  [MULTI_INTENT] Unknown intent in sequence: {step_intent}")
                        continue
                    
                    # Store the result
                    if step_result:
                        combined_responses.append({
                            "step": idx,
                            "intent": step_intent,
                            "content": step_result.get("content", ""),
                            "type": step_result.get("type", "response")
                        })
                        final_result = step_result  # Keep updating to get the last result
                
                # Return the final result (last step's result)
                if final_result:
                    # Add metadata about multi-intent execution
                    final_result["metadata"] = {
                        **final_result.get("metadata", {}),
                        **intent_metadata,
                        "multi_intent": True,
                        "steps_executed": len(combined_responses),
                        "intent_sequence": [s.get("intent") for s in intent_sequence]
                    }
                    
                    # Update last_bot_message
                    if 'content' in final_result:
                        session_data['last_bot_message'] = final_result['content'][:500]
                    
                    print(f"✅ [MULTI_INTENT] Completed {len(combined_responses)} steps successfully")
                    return final_result
                else:
                    # Fallback if no results
                    result = {
                        "type": "response",
                        "content": "I processed your request but encountered an issue. Please try breaking it into separate steps.",
                        "show_shortlist": False
                    }
                    result["metadata"] = {**result.get("metadata", {}), **intent_metadata}
                    return result


            if intent == "followup_customize":
                active = session_data.get("active_focus")
                if not active:
                    print(f"⚠️  [DEBUG] followup_customize but no active focus - generating lost context message")
                    clarification = self._generate_lost_context_clarification(user_input, session_data)
                    result = {
                        "type": "response",
                        "content": clarification,
                        "show_shortlist": len(session_data.get("last_results", [])) > 0,
                        "activities": session_data.get("last_results", [])[:8] if session_data.get("last_results") else []
                    }
                else:
                    result = self._handle_followup_customize(user_input, session_data)
                result["metadata"] = {**result.get("metadata", {}), **intent_metadata}
                if 'content' in result:
                    session_data['last_bot_message'] = result.get('content', result.get('chat_message', ''))[:500]
                return result
            
            if intent == "navigate":
                result = self._handle_navigate(user_input, session_data)
                result["metadata"] = {**result.get("metadata", {}), **intent_metadata}
                if 'content' in result:
                    session_data['last_bot_message'] = result['content'][:500]
                return result

            print(f"🔍 [DEBUG] Falling through to db_search (intent={intent})")
            result = self._handle_db_search(user_input, session_data)
            result["metadata"] = {**result.get("metadata", {}), **intent_metadata}
            
            # Store bot message for context
            if 'content' in result:
                session_data['last_bot_message'] = result['content'][:500]
                print(f"💾 [SESSION] Stored last_bot_message: {session_data['last_bot_message'][:100]}...")
                # Also store last_titles for context
                if result.get('activities'):
                    session_data['last_titles'] = [a.get('title', '') for a in result['activities'][:5]]
            
            return result

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"💥 [ERROR] process_message crashed:\n{error_trace}")

            return {
                "type": "response",
                "content": (
                    "I encountered an error processing your request. "
                    "Please try rephrasing or start a new search."
                ),
                "activities": [],
                "show_shortlist": False,
                "error": str(e),
                "metadata": {"error": str(e)}
            }

    def _handle_db_search(self, user_input: str, session_data: Dict) -> Dict[str, Any]:
        """Handle database search with query rewriting"""
        
        # Store raw input for fallback
        session_data["_raw_user_input"] = user_input
        
        query_obj = self.rewrite_to_db_query(user_input, session_data)
        
        print(f"🔍 [DEBUG] Query object: {query_obj}")
        
        results = self.run_db_search_pipeline(query_obj, session_data)
        if not results:
            results = self.no_results_strategy(query_obj, session_data)
        
        return self.render_search_answer(user_input, query_obj, results, session_data)
    
    def rewrite_to_db_query(self, user_input: str, session_data: Dict) -> Dict[str, Any]:
        """Convert user input into structured query object"""
        import json
        
        # Quick LLM query parsing
        prompt = f"""
Convert this user message into a search query JSON.

User message: "{user_input}"

Return ONLY JSON with these keys:
{{
    "topic": "main search term (1-3 words)",
    "synonyms": ["related", "terms"],
    "time_minutes": number or null,
    "constraints": ["specific", "requirements"]
}}

Examples:
- "find 15 minute activities" → {{"topic": "activities", "synonyms": [], "time_minutes": 15, "constraints": []}}
- "show me conflict activities" → {{"topic": "conflict", "synonyms": ["disagreement", "resolution"], "time_minutes": null, "constraints": []}}
- "learning names activities" → {{"topic": "learning names", "synonyms": ["name games", "introductions"], "time_minutes": null, "constraints": []}}
"""

        try:
            response = openai_service.chat(
                messages=[
                    {"role": "system", "content": "You convert user queries into structured JSON search queries."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0.1
            )
            
            query_obj = json.loads(response)
            print(f"🔍 [DEBUG] LLM parsed query: {query_obj}")
            
            return {
                "topic": query_obj.get("topic", ""),
                "synonyms": query_obj.get("synonyms", []),
                "grade_range": None,
                "time_minutes": query_obj.get("time_minutes"),
                "time_range": None,
                "constraints": query_obj.get("constraints", []),
                "must_have": [],
                "nice_to_have": []
            }
            
        except Exception as e:
            print(f"⚠️  [DEBUG] Query parsing failed: {e}")
            # Fallback: use raw input as topic
            return {
                "topic": user_input,
                "synonyms": [],
                "grade_range": None,
                "time_minutes": self._extract_minutes_from_text(user_input),
                "time_range": None,
                "constraints": [],
                "must_have": [],
                "nice_to_have": []
            }

    def run_db_search_pipeline(self, query_obj: Dict, session_data: Dict) -> List[Dict]:
        """
        Run the complete DB search pipeline with intelligent tiered sorting
        
        Strategy:
        1. Single semantic search by topic (30 results)
        2. Split into tiers by semantic score (≥0.7 = high relevance)
        3. Sort each tier by time proximity (if time specified)
        4. Combine tiers and return top 8
        """
        try:
            # ================================================================
            # STEP 1: Build search query from topic
            # ================================================================
            query_parts: List[str] = []

            topic = (query_obj.get("topic") or "").strip()
            if topic:
                query_parts.append(topic)

            synonyms = query_obj.get("synonyms") or []
            query_parts.extend([s.strip() for s in synonyms if s and s.strip()])

            constraints = query_obj.get("constraints") or []
            query_parts.extend([c.strip() for c in constraints if c and c.strip()])

            if query_parts:
                q = " ".join(query_parts)
            else:
                q = session_data.get("_raw_user_input", "activities")

            print(f"🔍 [DEBUG] Searching by topic: '{q}'")

            requested_time = query_obj.get("time_minutes")
            if requested_time:
                print(f"🔍 [DEBUG] Time preference: {requested_time} min")

            # ================================================================
            # STEP 2: Semantic search (get 30 results)
            # ================================================================
            distances, indices = data_processor.search(q, 30)
            all_results: List[Dict] = []

            for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
                try:
                    if idx < 0:
                        continue

                    row = data_processor.get_row(int(idx))
                    title = (data_processor.safe_get(row, "Strategic Action") or
                             data_processor.safe_get(row, "Title") or
                             "Untitled Activity")

                    time_str = (data_processor.safe_get(row, "Time") or
                                data_processor.safe_get(row, "Time to implement") or "").strip()

                    summary = (data_processor.safe_get(row, "Short Description") or
                               data_processor.safe_get(row, "Notes") or
                               data_processor.safe_get(row, "Directions") or "")

                    # Extract minutes
                    minutes = None
                    if time_str:
                        import re
                        m = re.search(r'\b(\d+)\s*(?:min|mins|minutes?)\b', time_str.lower())
                        if m:
                            minutes = int(m.group(1))

                    display_line = title
                    if time_str:
                        display_line += f" ⏱️ {time_str}"
                    if summary:
                        display_line += f" – {summary}"

                    all_results.append({
                        "row_index": int(idx),
                        "display_id": f"A{i}",
                        "title": title,
                        "time": time_str,
                        "time_str": time_str,
                        "summary": summary,
                        "minutes": minutes,
                        "score": float(dist),
                        "display_line": display_line
                    })

                except Exception as e:
                    print(f"⚠️  [DEBUG] Error processing result {idx}: {e}")
                    continue

            print(f"🔍 [DEBUG] Found {len(all_results)} total activities from search")

            # ================================================================
            # STEP 3: Split into tiers by semantic score
            # Tier 1: score ≥ 0.7 (high relevance)
            # Tier 2: score < 0.7 (lower relevance)
            # ================================================================
            TIER_THRESHOLD = 0.7
            
            tier1 = [r for r in all_results if r['score'] >= TIER_THRESHOLD]
            tier2 = [r for r in all_results if r['score'] < TIER_THRESHOLD]
            
            print(f"🔍 [DEBUG] Tier 1 (score ≥ {TIER_THRESHOLD}): {len(tier1)} activities")
            print(f"🔍 [DEBUG] Tier 2 (score < {TIER_THRESHOLD}): {len(tier2)} activities")

            # ================================================================
            # STEP 4: Sort each tier by time proximity (if time specified)
            # ================================================================
            def time_sort_key(activity):
                """
                Sort key prioritizing time match:
                1. Exact time match → score 0
                2. Close time (±5 min) → score 1-5
                3. Other times → score 100 + actual_time (ascending)
                4. No time info → score 999
                """
                activity_time = activity.get('minutes')
                
                if activity_time is None:
                    return 999
                
                if not requested_time:
                    # No time preference - sort by actual time ascending
                    return activity_time
                
                diff = abs(activity_time - requested_time)
                
                if diff == 0:
                    return 0  # Perfect match
                elif diff <= 5:
                    return diff  # Very close
                else:
                    return 100 + activity_time  # Sort by actual time ascending
            
            if requested_time:
                print(f"🔍 [DEBUG] Sorting both tiers by time proximity to {requested_time} min")
                tier1.sort(key=time_sort_key)
                tier2.sort(key=time_sort_key)
            else:
                # No time preference - keep semantic order (already sorted)
                print(f"🔍 [DEBUG] No time preference - keeping semantic order")

            # ================================================================
            # STEP 5: Combine tiers (Tier 1 first, then Tier 2)
            # ================================================================
            final_results = tier1 + tier2
            
            # Take top 8
            final_results = final_results[:8]
            
            # Fix display IDs to be sequential
            for i, r in enumerate(final_results, 1):
                r["display_id"] = f"A{i}"

            print(f"🔍 [DEBUG] Final {len(final_results)} activities:")
            for i, r in enumerate(final_results[:5], 1):
                score_label = "T1" if r['score'] >= TIER_THRESHOLD else "T2"
                print(f"  {i}. [{score_label}] {r['title']} - {r.get('time_str', 'no time')} (score: {r['score']:.2f})")

            session_data["last_results"] = final_results
            self.last_results = final_results

            return final_results

        except Exception as e:
            print(f"💥 [ERROR] Search pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    def no_results_strategy(self, query_obj: Dict, session_data: Dict) -> List[Dict]:
        return []
    

    def _format_search_results_with_llm(self, activities: List[Dict], user_query: str, session: Dict) -> str:
        """Use LLM to format search results contextually and naturally."""
        import json
        
        if len(activities) == 0:
            return "I couldn't find any activities matching that description. Could you try different keywords?"
        
        # Prepare activity data for LLM
        activity_summaries = []
        for i, act in enumerate(activities[:8], 1):
            activity_summaries.append({
                "number": i,
                "title": act.get('title', act.get('Title', 'Untitled')),
                "time": act.get('time_display') or act.get('time') or act.get('time_str') or act.get('Time', 'N/A'),
                "description": (act.get('summary') or act.get('description') or act.get('Description', ''))[:200],
                "similarity_score": round(act.get('score', 0), 2)
            })
        
        prompt = f"""You are helping a teacher find classroom activities. Format these search results naturally and helpfully.

Teacher's search: "{user_query}"
Number of results: {len(activities)}

Results with similarity scores (0.0-1.0, higher = better match):
{json.dumps(activity_summaries, indent=2)}

Instructions:
1. If only 1 result: Present it warmly and suggest they can type "yes" or "open it" to view the full lesson plan
2. If 1 result has significantly higher score than others (≥0.75): Highlight it as "Best match!" with 🎯 emoji and suggest opening it
3. If multiple similar scores: Present them equally as numbered options
4. Always show: number, title, time duration
5. Keep descriptions brief (1 line each)
6. Be conversational and encouraging
7. End with a helpful tip about how to select (e.g., "Type 'open 1' to view any activity")

DO NOT use JSON. Respond with ONLY the formatted message text that the teacher will see."""

        messages = [{"role": "user", "content": prompt}]
        
        try:
            formatted_response = openai_service.chat(
                messages=messages,
                model="gpt-4o-mini",
                temperature=0.4  # Balanced: consistent but friendly
            )
            
            print(f"✅ [LLM-FORMAT] Formatted {len(activities)} results contextually")
            return formatted_response
            
        except Exception as e:
            print(f"⚠️  [FORMAT-ERROR] LLM formatting failed: {e}, using fallback")
            return self._simple_activity_list(activities)

    def _simple_activity_list(self, activities: List[Dict]) -> str:
        """Simple fallback formatting if LLM fails."""
        lines = [f"Found {len(activities)} activities:\n\n"]
        for i, act in enumerate(activities[:8], 1):
            title = act.get('title', act.get('Title', 'Untitled'))
            time_info = act.get('time_display') or act.get('time') or act.get('time_str') or act.get('Time', 'N/A')
            description = (act.get('summary') or act.get('description') or act.get('Description', ''))[:150]
            lines.append(
                f"{i}. **{title}** ⏱️ {time_info}\n"
                f"   {description}...\n"
            )
        lines.append(f"\nType 'open' followed by a number to view an activity.")
        return "".join(lines)

    def _generate_lost_context_clarification(self, user_message: str, session: Dict) -> str:
        """Use LLM to generate contextual clarification when user tries to modify without open activity."""
        import json
        
        recent_searches = session.get('last_titles', [])
        if not recent_searches:
            recent_searches = [r.get('title', '') for r in session.get('last_results', [])[:3]]
        
        prompt = f"""A teacher wants to modify an activity, but no activity is currently open.

Teacher's request: "{user_message}"
Recent searches: {recent_searches[:3] if recent_searches else "No recent searches"}

Generate a helpful, friendly message that:
1. Acknowledges what they want to do
2. Explains no activity is currently open
3. Suggests concrete next steps (search for an activity OR specify which recent result)
4. Keep it brief (2-3 sentences max)
5. Be encouraging and helpful
6. IMPORTANT: End with a question to prompt the user (e.g., "Which activity would you like to modify?")

Respond with ONLY the message text, no JSON."""

        messages = [{"role": "user", "content": prompt}]
        
        try:
            clarification = openai_service.chat(
                messages=messages,
                model="gpt-4o-mini",
                temperature=0.3
            )
            print(f"✅ [CLARIFICATION] Generated contextual help message")
            return clarification
            
        except Exception as e:
            print(f"⚠️  [CLARIFICATION-ERROR] LLM failed: {e}, using fallback")
            return "I don't have an activity open right now. Could you search for an activity first, or tell me which one you'd like to modify?"

    def render_search_answer(self, user_input: str, query_obj: Dict, results: List[Dict], session_data: Dict) -> Dict[str, Any]:
        """Render search results with full descriptions (no truncation)"""
        if not results:
            return {
                "type": "response",
                "content": "I couldn't find any activities that match your request.",
                "activities": [],
                "show_shortlist": False
            }

        # Enrich activities with resource links if available
        for activity in results:
            row_index = activity.get('row_index')
            if row_index is not None:
                try:
                    row = data_processor.get_row(row_index)
                    resource_link = (data_processor.safe_get(row, "Link to Resource") or 
                                     data_processor.safe_get(row, "Link to the Source Materials") or
                                     data_processor.safe_get(row, "Resources Needed"))
                    activity['resource_link'] = resource_link
                except Exception as e:
                    print(f"⚠️  [DEBUG] Error getting resource link for row {row_index}: {e}")
                    activity['resource_link'] = None

        # Clear stale modification state AND active focus on new search
        session_data['last_modified_activity'] = None
        session_data['last_modification_history'] = []
        session_data['active_focus'] = None  # ← ADD THIS LINE
        session_data['last_opened_index'] = None  # ← ADD THIS LINE
        print(f"🔄 [SEARCH] Cleared stale modification state AND active focus")
        
        # Persist last results
        session_data["last_results"] = results
        session_data["last_query"] = query_obj
        self.last_results = results
        for activity in results:
            activity_id = activity.get('row_index') or activity.get('id')
            if activity_id:
                self.all_activities_seen[activity_id] = activity

        # Format results with LLM
        response_content = self._format_search_results_with_llm(
            results, 
            user_input,
            session_data
        )
        
        # Store titles for context
        session_data['last_titles'] = [r.get('title', '') for r in results[:5]]

        print(f"📊 [DEBUG] Returning {len(results)} activities with LLM-formatted descriptions")

        return {
            "type": "response",
            "content": response_content,
            "activities": results,
            "show_shortlist": True,
            "metadata": {
                "results_count": len(results),
                "detected_intent": "db_search"
            }
        }

    def _handle_followup_customize(self, user_input: str, session_data: Dict) -> Dict[str, Any]:
        """Handle customization requests using structured activity system with source awareness"""
        # Import at top of method to avoid scoping issues
        from processing.document_processor import document_processor
        
        active_focus = session_data.get("active_focus", {})
        if not active_focus:
            return self._handle_db_search(user_input, session_data)
        
        # Quick intelligence check: Does this REALLY need modification?
        # If user is asking WHAT/HOW/WHY without clear modification intent, redirect to question handler
        user_lower = user_input.lower()
        
        # Check if this is actually a question, not a modification request
        question_indicators = ['what is', 'how does', 'why does', 'explain', 'tell me about', 'what does', 'how do i use']
        modification_indicators = ['add', 'give', 'provide', 'change', 'make', 'create', 'need', 'include', 'modify', 'update']
        
        has_question_words = any(word in user_lower for word in question_indicators)
        has_modification_words = any(word in user_lower for word in modification_indicators)
        
        # If it's clearly a question without modification intent, redirect
        if has_question_words and not has_modification_words:
            # But check if they're asking for content to be added (e.g., "what statements should I use?")
            if not any(phrase in user_lower for phrase in ['should i', 'can you', 'give me', 'i need', 'provide']):
                print(f"🔍 [REDIRECT] This looks more like a question than customization - redirecting to followup_question")
                return self._handle_followup_question(user_input, session_data)

        # ✅ FIX: Pronoun resolution using last_modification context
        if "those" in user_lower or "these" in user_lower or "them" in user_lower:
            last_mod = session_data.get('last_modification')
            if last_mod:
                mod_type = last_mod.get('type')
                field = last_mod.get('field')
                count = last_mod.get('count')
                
                # Expand pronoun to explicit reference
                if mod_type == 'modify_questions' and field == 'Reflection Questions':
                    if "remove" in user_lower or "delete" in user_lower:
                        user_input = user_input.replace("those", f"the last {count or 2} reflection questions")
                        user_input = user_input.replace("these", f"the last {count or 2} reflection questions")
                        user_input = user_input.replace("them", f"the last {count or 2} reflection questions")
                        print(f"🔍 [PRONOUN] Resolved to: {user_input}")
                elif mod_type == 'shorten' and field == 'Directions':
                    if "expand" in user_lower or "lengthen" in user_lower:
                        user_input = user_input.replace("those", "the directions")
                        user_input = user_input.replace("these", "the directions")
                        user_input = user_input.replace("them", "the directions")
                        print(f"🔍 [PRONOUN] Resolved to: {user_input}")
    
        # ====================================================================
        # NEW: CHECK IF REQUESTED CONTENT ALREADY EXISTS
        # Prevent unnecessary full rewrites when user just wants small additions
        # ====================================================================
        # Use previously modified activity if it exists
        # CRITICAL: We need the PARSED activity dict (with Directions, Materials, etc.)
        # not the raw CSV row data
        if session_data.get("last_modified_activity"):
            base_activity = session_data["last_modified_activity"]
            print(f"🔍 [DEBUG] Using MODIFIED activity for customization")
            
            # ✅ VALIDATE: Check if MODIFIED is broken
            if not base_activity.get("Directions"):
                print(f"⚠️ [DEBUG] MODIFIED activity missing Directions, restoring from pre-generated")
                
                if session_data.get("last_opened_index") is not None:
                    row_index = session_data["last_opened_index"]
                    lesson_plan_md, summary = content_cache.get_generated(row_index)
                    
                    if lesson_plan_md:
                        base_activity = document_processor.markdown_to_dict(lesson_plan_md)
                        print(f"✅ [DEBUG] Restored from pre-generated cache: {list(base_activity.keys())}")
                    else:
                        return {
                            "type": "response",
                            "content": "Cannot modify - the activity data is corrupted. Please re-open the activity first.",
                            "show_shortlist": False
                        }
                else:
                    return {
                        "type": "response",
                        "content": "Cannot modify - no activity is currently open. Please open an activity first.",
                        "show_shortlist": False
                    }
            
            # ✅ CRITICAL: Set activity to the validated/restored base_activity
            activity = base_activity
        else:
            # First modification - use pre-generated markdown
            if session_data.get("last_opened_index") is not None:
                row_index = session_data["last_opened_index"]
                
                # ✅ Get pre-generated markdown from CONTENT CACHE (not DataFrame!)
                lesson_plan_md, summary = content_cache.get_generated(row_index)
                
                if lesson_plan_md:
                    base_activity = document_processor.markdown_to_dict(lesson_plan_md)
                    print(f"🔍 [DEBUG] Using ORIGINAL pre-generated cache")
                    activity = base_activity
                else:
                    # Fallback to parsing content_md
                    content_md = active_focus.get("content_md", "")
                    if content_md:
                        activity = document_processor.markdown_to_dict(content_md)
                        print(f"🔍 [DEBUG] Parsed activity from markdown: {list(activity.keys())}")
                    else:
                        # Fallback to raw activity (will likely fail)
                        activity = active_focus.get("activity", {})
                        print(f"🔍 [DEBUG] Using raw CSV data (no pre-generated available)")
            else:
                # Parse the content_md to get structured activity dict
                content_md = active_focus.get("content_md", "")
                if content_md:
                    activity = document_processor.markdown_to_dict(content_md)
                    print(f"🔍 [DEBUG] Parsed activity from markdown: {list(activity.keys())}")
                else:
                    # Fallback to raw activity (will likely fail)
                    activity = active_focus.get("activity", {})
                    print(f"⚠️  [DEBUG] Using raw activity (no markdown available)")
        
        # ✅ Now activity is guaranteed to have Directions
        print(f"🔍 [DEBUG] Activity for modification has keys: {list(activity.keys())}")
        print(f"🔍 [DEBUG] Activity has Directions: {bool(activity.get('Directions'))}")
        print(f"🔍 [DEBUG] Directions length: {len(activity.get('Directions', ''))}")
        
        content_md = active_focus.get("content_md", "")
        
        # ====================================================================
        # USE FULL LLM MODIFICATION FOR ALL REQUESTS
        # This handles ANY customization intelligently (numbers, examples, etc.)
        # ====================================================================
        try:
            EXPAND_RE = re.compile(r"\b(expand|lengthen|elaborate|add (detail|details)|make .*longer|add (steps?|examples?))\b", re.I)
            SHORTEN_RE = re.compile(r"\b(shorten|condense|summariz(e|e it)|make .*shorter|trim)\b", re.I)
            
            # Get row_index from multiple possible sources
            row_index = active_focus.get("row_index") or session_data.get("last_opened_index")
            
            # ✅ FIX: Check if activity is actually open
            if row_index is None:
                return {
                    "type": "response",
                    "content": "I don't have an activity open right now. Could you search for an activity first, or let me know which one you'd like to modify?",
                    "activities": session_data.get('last_results', [])[:8],
                    "show_shortlist": bool(session_data.get('last_results')),
                    "debug": {"note": "No active activity for modification"}
                }

            if not activity:
                return self._handle_db_search(user_input, session_data)
            
            print(f"🔍 [DEBUG] Full modification for: {user_input[:100]}")

            # ====================================================================
            # Get source content for context-aware modifications
            # ====================================================================
            source_content = None
            if row_index is not None:
                try:
                    row_data = data_processor.get_row(row_index)
                    source_content, _ = content_retrieval.text_from_supabase_or_fallback(row_data, allow_web=True)

                    if source_content and len(source_content) > 100:
                        print(f"✅ [DEBUG] Retrieved source content: {len(source_content)} chars")
                    else:
                        print(f"⚠️  [DEBUG] No substantial source content available")
                        source_content = None

                except Exception as e:
                    print(f"⚠️  [DEBUG] Could not retrieve source content: {e}")
                    source_content = None

            # CRITICAL FIX: For expand operations, use the PARSED original
            # For other operations, use current activity (which could be modified)
            original_activity = active_focus.get("original_activity")

            # ✅ ALWAYS restore original_activity from cache since it's stored as raw CSV
            if session_data.get("last_opened_index") is not None:
                row_index = session_data["last_opened_index"]
                lesson_plan_md, summary = content_cache.get_generated(row_index)
                if lesson_plan_md:
                    original_activity = document_processor.markdown_to_dict(lesson_plan_md)
                    print(f"✅ [DEBUG] Loaded original_activity from cache: {list(original_activity.keys())}")
                else:
                    # Fallback to using validated activity
                    print(f"⚠️ [DEBUG] No cache available for original, using validated activity")
                    original_activity = activity
            else:
                # Fallback to using validated activity
                print(f"⚠️ [DEBUG] No last_opened_index for original, using validated activity")
                original_activity = activity

            text_lc = (user_input or "").lower()
            is_expand = bool(EXPAND_RE.search(text_lc))
            is_shorten = bool(SHORTEN_RE.search(text_lc))
            # ✅ Use validated activities - both should have Directions now
            base_for_modify = original_activity if is_expand else activity

            # ✅ Store original activity in session for chat message generation
            session_data["original_activity_cache"] = original_activity.copy()


            # ====================================================================
            # CALL FULL MODIFICATION - with source_content parameter
            # ====================================================================
            # Use row-based modification with smart modifier
            conversation_history = session_data.get("messages", [])

            # Debug: Print last few messages
            print("🔍 [DEBUG] Last 3 messages in session:")
            for i, msg in enumerate(session_data.get("messages", [])[-3:]):
                print(f"  {i}: {msg.get('role')}: {msg.get('content', '')[:100]}")

            result = document_processor.modify_activity(
                row_index=row_index,
                instruction=user_input,
                conversation_history=conversation_history
            )

            # ✅ Handle clarification requests (returns dict)
            if isinstance(result, dict) and result.get("clarification_needed"):
                return {
                    "type": "response",
                    "content": result.get("clarification_message"),
                    "chat_message": result.get("clarification_message"),
                    "view": "chat",
                    "show_shortlist": False
                }

            # Normal flow: result is a tuple (formatted_md, message)
            formatted_md, _ = result

            # Parse the formatted markdown back to dict
            modified = document_processor.markdown_to_dict(formatted_md)

            # ====================================================================
            # STORE MODIFIED ACTIVITY IN SESSION FOR NEXT MODIFICATION
            # CRITICAL: Store the PARSED activity dict (with Directions, Materials, etc.)
            # ====================================================================
            session_data["active_focus"]["modified_activity"] = modified
            session_data["last_modified_activity"] = modified

            # ✅ FIX: Track last modification details for pronoun resolution
            import time
            mod_type = 'generic'
            field = None
            count = None
            
            user_lower = user_input.lower()
            if 'question' in user_lower:
                mod_type = 'modify_questions'
                field = 'Reflection Questions'
                # Try to extract count
                numbers = re.findall(r'\d+', user_input)
                if numbers:
                    count = int(numbers[0])
            elif 'shorten' in user_lower or 'condense' in user_lower:
                mod_type = 'shorten'
                field = 'Directions'
            elif 'expand' in user_lower or 'lengthen' in user_lower:
                mod_type = 'expand'
                field = 'Directions'
            
            session_data['last_modification'] = {
                'type': mod_type,
                'field': field,
                'count': count,
                'timestamp': time.time()
            }
            print(f"💾 [SESSION] Tracked last modification: {mod_type} on {field}")
            
            # Also update content_md so we can parse it later
            session_data["active_focus"]["content_md"] = document_processor.legacy_to_markdown(modified)
            print(f"💾 [SESSION] Stored modified activity for future modifications")
            print(f"🔍 [SESSION] Modified activity keys: {list(modified.keys())[:5]}...")

            # ====================================================================
            # ENFORCE TITLE PRESERVATION (Triple-check)
            # ====================================================================
            original_title = (active_focus.get("title") or
                              activity.get("Activity Name") or
                              activity.get("Title"))
            if original_title:
                modified["Activity Name"] = original_title
                if "Title" in modified:
                    modified["Title"] = original_title
                session_data["active_focus"]["title"] = original_title
                print(f"✅ [ENGINE] Title after modification: '{modified.get('Activity Name')}'")
            
            print(f"🔍 [DEBUG] Modified activity has {len(modified)} fields")
            
            title = (modified.get("Activity Name") or modified.get("Title") or
                     session_data.get("active_focus", {}).get("title") or "Lesson Plan")
            
            session_data["active_focus"]["activity"] = modified
            session_data["active_focus"]["title"] = title
            
            md = document_processor.legacy_to_markdown(modified)
            
            session_data["active_focus"]["content_md"] = md
            
            from base64 import b64encode
            try:
                doc_bytes = document_processor.build_docx(modified)
            except Exception as e:
                print(f"⚠️  build_docx failed: {e}")
                doc_bytes = document_processor.build_plain_docx(md)
            doc_bytes_base64 = b64encode(doc_bytes).decode("utf-8")
            
            filename = f"{(title or 'Lesson Plan').replace('/', '-')}.docx"
            
            # Build chat message with modification metadata
            modification_type = session_data.get("last_modification", {}).get("type", "unknown")
            modification_field = session_data.get("last_modification", {}).get("field", "unknown")

            if modification_type == "shorten":
                # Get the actual lengths from the activities
                original_activity = session_data.get("original_activity_cache", {})
                original_len = len(original_activity.get("Directions", ""))
                modified_len = len(modified.get("Directions", ""))
                reduction_pct = ((original_len - modified_len) / original_len * 100) if original_len > 0 else 0
                
                chat_msg = f"I've shortened the directions from {original_len} to {modified_len} characters ({reduction_pct:.1f}% reduction)."
            elif modification_type == "modify_questions":
                original_activity = session_data.get("original_activity_cache", {})
                original_count = len([q for q in original_activity.get("Reflection Questions", "").split('\n') if q.strip().startswith('-')])
                modified_count = len([q for q in modified.get("Reflection Questions", "").split('\n') if q.strip().startswith('-')])
                
                chat_msg = f"I've modified the reflection questions (changed from {original_count} to {modified_count} questions)."
            elif modification_type == "generic":
                chat_msg = f"I've modified the activity as requested."
            else:
                chat_msg = f"Here's the modified activity."

            return {
                "type": "lesson_plan",
                "content": md,
                "content_md": md,
                "chat_message": chat_msg,  # ← ADD THIS - will be appended to conversation history
                "lesson_plan": {
                    "title": title,
                    "filename": filename,
                    "activity": modified,
                    "row_index": row_index,
                    "doc_bytes": doc_bytes_base64
                },
                "view": "lesson_plan",
                "title": title,
                "show_shortlist": False
            }
            
        except Exception as e:
            print(f"💥 [ERROR] _handle_followup_customize: {e}")
            import traceback
            traceback.print_exc()
            return {
                "type": "response",
                "content": f"I encountered an error modifying the activity: {str(e)}. Please try rephrasing your request.",
                "show_shortlist": False
            }
    def _handle_followup_question(self, user_input: str, session_data: Dict) -> Dict[str, Any]:
        """Handle questions about the currently active activity"""
        active_focus = session_data.get("active_focus", {})
        if not active_focus:
            return {
                "type": "response",
                "content": "Please open an activity first, then I can answer questions about it.",
                "show_shortlist": False
            }
            
        activity = active_focus.get("activity", {})
        title = active_focus.get("title", "this activity")
        
        activity_context = f"""
Activity: {title}

Activity Details:
- Objective: {activity.get('Objective', 'N/A')}
- Materials: {activity.get('Materials', 'N/A')}
- Time: {activity.get('Time', 'N/A')}
- Directions: {activity.get('Directions', 'N/A')[:500]}...
"""
        
        prompt = f"""You are an experienced educator helping a teacher understand and use a classroom activity.

{activity_context}

Teacher's question: "{user_input}"

Provide helpful, specific answers about this activity:
1. Answer their question directly with concrete information from the activity
2. Include specific examples or steps from the activity
3. Explain how to implement or adapt the activity based on their question
4. Keep it conversational and supportive (2-3 paragraphs max)
5. If they ask about modifications, suggest specific changes they can make

Be practical and reference the activity details above.
"""
        
        try:
            from services.ai_services import openai_service
            response = openai_service.chat(
                messages=[
                    {"role": "system", "content": "You are a supportive educator helping teachers understand and use classroom activities."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return {
                "type": "response",
                "content": response,
                "show_shortlist": False,
                "debug": {"intent": "followup_question", "activity": title}
            }
            
        except Exception as e:
            print(f"💥 [ERROR] _handle_followup_question: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "type": "response",
                "content": f"I encountered an error answering your question about {title}. Please try rephrasing.",
                "show_shortlist": False
            }
    
    def _handle_navigate(self, user_input: str, session_data: Dict) -> Dict[str, Any]:
        """Handle navigation requests"""
        return self._handle_db_search(user_input, session_data)

    def _handle_acknowledgment(self, user_input: str, routed: Dict, session_data: Dict) -> Dict[str, Any]:
        """Handle acknowledgment messages dynamically using LLM (FIXED: Less verbose)"""
        active_focus = session_data.get("active_focus")
        metadata = routed.get("metadata", {}) or {}
        active_title = metadata.get("active_title") or (active_focus.get("title") if active_focus else None)

        # Case 1: Acknowledging current active activity
        if active_focus and active_title:
            row_index = active_focus.get("row_index")
            prompt = f"""User: "{user_input}"

They're satisfied with "{active_title}".

Respond in ONE sentence. Be brief and friendly.
Mention they can download or modify it."""

            try:
                response_text = openai_service.chat(
                    messages=[
                        {"role": "system", "content": "You are a helpful educational assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50,  # REDUCED FROM 150
                    temperature=0.7
                )
                return {
                    "type": "response",
                    "content": response_text,
                    "show_shortlist": False,
                    "metadata": {
                        "acknowledgment_type": "active_activity",
                        "active_activity_id": row_index,
                        "active_activity_title": active_title
                    }
                }
            except Exception as e:
                print(f"⚠️  [ACKNOWLEDGMENT] LLM error: {e}")
                return {
                    "type": "response",
                    "content": f"Great! You can download {active_title} now or let me know if you need any changes.",
                    "show_shortlist": False,
                    "metadata": {
                        "acknowledgment_type": "active_activity",
                        "active_activity_id": row_index,
                        "active_activity_title": active_title
                    }
                }

        # Case 2: Generic acknowledgment
        prompt = f"""User: "{user_input}"

Respond in ONE sentence. Be brief and friendly."""

        try:
            response_text = openai_service.chat(
                messages=[
                    {"role": "system", "content": "You are a helpful educational assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.7
            )
            content = response_text
        except Exception as e:
            print(f"⚠️  [ACKNOWLEDGMENT] LLM error: {e}")
            content = "You're welcome! Let me know if you need anything else."

        return {
            "type": "response",
            "content": content,
            "show_shortlist": False,
            "metadata": {
                "acknowledgment_type": "generic"
            }
        }
    
    def _handle_open_activity(self, user_input: str, routed: Dict, session_data: Dict) -> Dict[str, Any]:
        """Handle opening an activity from last results"""
        
        # 🧪 TEMPORARY: Force on-demand generation to test new formatting
        FORCE_REGENERATE = False  # ← Force skip cache to test new formatting
        
        print(f"🔍 [DEBUG] db_only: Handling open_activity")
        
        if routed.get("target") == "disambiguate" and routed.get("options"):
            return {
                "type": "response",
                "content": "Did you mean one of these activities?",
                "options": routed["options"],
                "show_shortlist": False
            }
        
        last = session_data.get("last_results") or []
        title = routed.get("title")
        payload = routed.get("payload") or {}

        row_index = payload.get("row_index") or payload.get("id")
        
        # ✅ FIX: Validate index bounds
        if row_index is not None:
            if row_index < 1:
                return {
                    "type": "response",
                    "content": "That's not a valid activity number. Please choose from the activities I showed you (1-8).",
                    "activities": session_data.get('last_results', [])[:8],
                    "show_shortlist": True
                }
            
            # Check if index is in last_results
            last_indices = [item.get('row_index') or item.get('id') for item in session_data.get('last_results', [])]
            if last_indices and row_index not in last_indices:
                return {
                    "type": "response",
                    "content": f"I don't see activity #{row_index} in the current results. Please choose from activities 1-{len(last_indices)}.",
                    "activities": session_data.get('last_results', [])[:8],
                    "show_shortlist": True
                }
        
        # Try to extract row index from target like "activity 14"
        if row_index is None:
            target = routed.get("target", "")
            if target:
                # Try "activity 14" format
                match = re.search(r"activity\s+(\d+)", str(target), re.IGNORECASE)
                if match:
                    row_index = int(match.group(1))
                    print(f"🔍 [DEBUG] Extracted row_index={row_index} from target={target}")
                    
                    # ✅ FIX: Validate extracted index
                    if row_index < 1:
                        return {
                            "type": "response",
                            "content": "That's not a valid activity number. Please choose from the activities I showed you (1-8).",
                            "activities": session_data.get('last_results', [])[:8],
                            "show_shortlist": True
                        }
                    
                    # Check if index is in last_results
                    last_indices = [item.get('row_index') or item.get('id') for item in session_data.get('last_results', [])]
                    if last_indices and row_index not in last_indices:
                        return {
                            "type": "response",
                            "content": f"I don't see activity #{row_index} in the current results. Please choose from activities 1-{len(last_indices)}.",
                            "activities": session_data.get('last_results', [])[:8],
                            "show_shortlist": True
                        }

        if row_index is None and title:
            for li in last:
                if (li.get("title") or li.get("Strategic Action")) == title:
                    row_index = li.get("row_index")
                    payload = li
                    break

        if row_index is None:
            return {
                "type": "response",
                "content": "I couldn't find that item in the current list.",
                "activities": [],
                "show_shortlist": False
            }
        
        try:
            row = data_processor.get_row(row_index)
        except Exception as e:
            print(f"🔍 [DEBUG] db_only: Error getting row: {e}")
            return {
                "type": "response",
                "content": f"Could not load activity data for **{title or 'this item'}**.",
                "activities": [],
                "show_shortlist": False
            }

        plain_activity = to_plain_activity(row)

        # Extract metadata
        title_final = (data_processor.safe_get(row, "Strategic Action") or 
                      data_processor.safe_get(row, "Activity Name") or 
                      data_processor.safe_get(row, "Title") or 
                      "Untitled Activity")

        # Extract resource links
        resource_link = (data_processor.safe_get(row, "Link to Resource") or 
                        data_processor.safe_get(row, "Link to the Source Materials") or
                        data_processor.safe_get(row, "Resources Needed"))
        
        reference_link = data_processor.safe_get(row, "Reference Link")

        # ====================================================================
        # TRY PRE-GENERATED CONTENT FIRST (INSTANT!)
        # ====================================================================
        
        lesson_plan_md, summary = content_cache.get_generated(row_index)
        
        # 🔍 DEBUG: Print what we got
        print(f"🔍 [DEBUG] lesson_plan_md type: {type(lesson_plan_md)}")
        print(f"🔍 [DEBUG] lesson_plan_md value: {repr(lesson_plan_md[:100]) if lesson_plan_md else 'None'}")
        print(f"🔍 [DEBUG] lesson_plan_md truthiness: {bool(lesson_plan_md)}")
        print(f"🔍 [DEBUG] FORCE_REGENERATE: {FORCE_REGENERATE}")
        print(f"🔍 [DEBUG] Condition result: {bool(lesson_plan_md and not FORCE_REGENERATE)}")
        
        
        if lesson_plan_md and not FORCE_REGENERATE:  # Don't require summary
            # 🚀 USE PRE-GENERATED CONTENT (0.1s)
            print(f"⚡ [INSTANT OPEN] Using pre-generated content for row {row_index}")
            
            # Parse the lesson plan markdown to extract all fields
            parsed_activity = document_processor.markdown_to_dict(lesson_plan_md)
            
            # Ensure title is set correctly (override any title from markdown with the actual title)
            parsed_activity["Activity Name"] = title_final
            parsed_activity["Title"] = title_final
            if "Strategic Action" in plain_activity:
                parsed_activity["Strategic Action"] = title_final
            
            # Merge with plain_activity to preserve any additional fields
            sanitized_activity = {**plain_activity, **parsed_activity}
            
            # Generate filename
            title_safe = "".join(c for c in title_final if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = f"{title_safe}.docx"
            
            # Store in session
            self.active_activity = sanitized_activity
            session_data["last_opened_index"] = row_index
            session_data["active_focus"] = {
                "activity_id": row_index,
                "row_index": row_index,
                "title": title_final,
                "activity": sanitized_activity,  # PARSED activity with Directions, Materials, etc.
                "content_md": lesson_plan_md,    # Store markdown for re-parsing if needed
                "row_data": plain_activity,
                "ts": __import__("time").time(),
                "embedding": None,
            }

            if row_index not in self.all_activities_seen:
                self.all_activities_seen[row_index] = {
                    "row_index": row_index,
                    "id": row_index,
                    "title": title_final,
                    "activity": sanitized_activity
                }

            if "original_activity" not in session_data["active_focus"]:
                session_data["active_focus"]["original_activity"] = copy.deepcopy(plain_activity)

            # Generate docx from markdown
            from base64 import b64encode
            try:
                # Try to build docx from the markdown
                # Create a temporary activity dict for docx generation
                temp_activity = {"Activity Name": title_final, "Content": lesson_plan_md}
                doc_bytes = document_processor.build_docx(temp_activity)
            except Exception:
                # Fallback to plain docx
                doc_bytes = document_processor.build_plain_docx(lesson_plan_md)
            doc_bytes_base64 = b64encode(doc_bytes).decode("utf-8")

            return {
                "type": "lesson_plan",
                "content": summary + "\n\n" + lesson_plan_md,
                "content_md": lesson_plan_md,
                "lesson_plan": {
                    "title": title_final,
                    "filename": filename,
                    "activity": sanitized_activity,
                    "row_index": int(row_index),
                    "doc_bytes": doc_bytes_base64,
                    "resource_link": resource_link,
                    "reference_link": reference_link
                },
                "view": "lesson_plan",
                "title": title_final,
                "show_shortlist": False
            }
        
        # ====================================================================
        # FALLBACK: GENERATE ON-DEMAND (24s)
        # ====================================================================
        
        print(f"🔄 [GENERATING] No pre-generated content, generating fresh for row {row_index}")
        
        # Get raw source from cache
        source_content, cache_status = content_cache.get(row_index)
        
        if cache_status != 'success' or not source_content:
            print(f"❌ [ERROR] No source content for row {row_index}")
            return {
                "type": "response",
                "content": "Sorry, I couldn't load that activity. The source content is missing.",
                "activities": [],
                "show_shortlist": False
            }
        
        print(f"✅ [CACHE HIT] Retrieved source: {len(source_content)} chars")

        text, fetch_error = content_retrieval.text_from_supabase_or_fallback(row, allow_web=True)
        
        if (text or "").strip():
            directions = document_processor.make_directions(text, title_final)
        else:
            directions = "Directions not available."

        block, filename, _, activity = document_processor.format_activity_block(row, directions)

        full_markdown = document_processor.legacy_to_markdown(activity)

        from utils.serialization import strip_binary_fields
        sanitized_activity = strip_binary_fields(activity)

        self.active_activity = sanitized_activity

        session_data["last_opened_index"] = row_index
        session_data["active_focus"] = {
            "activity_id": row_index,
            "row_index": row_index,
            "title": title_final,
            "activity": plain_activity,
            "row_data": plain_activity,
            "ts": __import__("time").time(),
            "embedding": None,
        }

        if row_index not in self.all_activities_seen:
            self.all_activities_seen[row_index] = {
                "row_index": row_index,
                "id": row_index,
                "title": title_final,
                "activity": sanitized_activity
            }

        if "original_activity" not in session_data["active_focus"]:
            session_data["active_focus"]["original_activity"] = copy.deepcopy(plain_activity)

        # After generating, cache it for next time!
        print(f"💾 Caching generated content for future use...")
        # Generate a summary for caching
        try:
            summary_for_cache = openai_service.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly teaching assistant. Write a brief, conversational 2-3 sentence introduction to this activity."
                    },
                    {
                        "role": "user",
                        "content": f"""Activity:
{full_markdown[:800]}

Write a warm, encouraging introduction (2-3 sentences max)."""
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            content_cache.set_generated(row_index, full_markdown, summary_for_cache)
        except Exception as e:
            print(f"⚠️  Could not cache generated content: {e}")

        from base64 import b64encode
        try:
            doc_bytes = document_processor.build_docx(activity)
        except Exception:
            doc_bytes = document_processor.build_plain_docx(block)
        doc_bytes_base64 = b64encode(doc_bytes).decode("utf-8")

        return {
            "type": "lesson_plan",
            "content": full_markdown,
            "content_md": full_markdown,
            "lesson_plan": {
                "title": title_final,
                "filename": filename,
                "activity": sanitized_activity,
                "row_index": int(row_index),
                "doc_bytes": doc_bytes_base64,
                "resource_link": resource_link,
                "reference_link": reference_link
            },
            "view": "lesson_plan",
            "title": title_final,
            "show_shortlist": False
        }



    def _execute_multi_step(self, steps: List[Dict], session: Dict, original_message: str) -> Dict:
        """
        Execute multiple intents in sequence.
        Each step builds on the previous one's session state.
        """
        
        total_steps = len(steps)
        print(f"🔄 [MULTI-STEP] Executing {total_steps} steps sequentially")
        
        results = []
        final_response = None
        
        for i, step in enumerate(steps, 1):
            action = step.get("action", "unknown")
            description = step.get("description", "")
            target = step.get("target", "")
            
            print(f"📍 [STEP {i}/{total_steps}] Action: {action}, Description: {description}")
            
            try:
                if action == "search":
                    # Execute search
                    final_response = self._handle_db_search(description, session)
                    
                    if final_response.get("type") == "response":
                        results.append(f"Found activities")
                    else:
                        results.append(f"Search completed")
                        
                elif action == "open":
                    # Execute open - need to map target to activity
                    
                    # First, check if we have recent search results
                    if not session.get("last_results"):
                        print(f"⚠️ [STEP {i}] Cannot open - no search results available")
                        results.append(f"Skipped open (no results)")
                        continue
                    
                    # Map target to activity reference and find the activity
                    activity_ref = self._map_target_to_activity(target, session)
                    
                    if not activity_ref:
                        print(f"⚠️ [STEP {i}] Could not resolve target: {target}")
                        results.append(f"Skipped open (invalid target)")
                        continue
                    
                    # Find the actual activity from last_results
                    activity_index = None
                    activity_payload = None
                    if activity_ref.startswith("A"):
                        try:
                            activity_index = int(activity_ref[1:]) - 1  # Convert A1 -> 0, A2 -> 1, etc.
                            if 0 <= activity_index < len(session.get("last_results", [])):
                                activity_payload = session["last_results"][activity_index]
                        except (ValueError, IndexError):
                            pass
                    
                    # Create synthetic route result for open
                    open_route = {
                        "intent": "open_activity",
                        "target": activity_ref,
                        "confidence": 0.95,
                        "source": "multi_step",
                        "payload": activity_payload or {},
                        "row_index": activity_payload.get("row_index") if activity_payload else None,
                        "title": activity_payload.get("title") if activity_payload else None
                    }
                    
                    final_response = self._handle_open_activity(description, open_route, session)
                    results.append(f"Opened activity")
                    
                elif action == "modify":
                    # Execute modification
                    if session.get("last_opened_index") is None:
                        print(f"⚠️ [STEP {i}] Cannot modify - no activity open")
                        results.append(f"Skipped modify (no activity open)")
                        continue
                    
                    # Create synthetic route result for modify
                    modify_route = {
                        "intent": "followup_customize",
                        "confidence": 0.95,
                        "source": "multi_step"
                    }
                    
                    final_response = self._handle_followup_customize(description, session)
                    results.append(f"Modified activity")
                
                else:
                    print(f"⚠️ [STEP {i}] Unknown action: {action}")
                    results.append(f"Skipped unknown action")
                    
            except Exception as e:
                print(f"❌ [STEP {i}] Error: {e}")
                results.append(f"Error in step {i}")
                import traceback
                traceback.print_exc()
        
        # Return the final step's response with summary
        if final_response:
            summary = " → ".join(results)
            
            # Add summary based on response type
            if final_response.get("type") == "response":
                # For search results, prepend summary
                content = final_response.get("content", "")
                final_response["content"] = f"✅ Completed: {summary}\n\n{content}"
                
            elif final_response.get("type") == "lesson_plan":
                # For lesson plans, add summary to chat_message
                chat_msg = final_response.get("chat_message", "")
                final_response["chat_message"] = f"✅ Completed: {summary}\n\n{chat_msg}"
            
            return final_response
        
        # Fallback if no final response
        return {
            "type": "response",
            "content": f"✅ Completed steps: {' → '.join(results)}",
            "chat_message": f"✅ Completed steps: {' → '.join(results)}",
            "view": "chat"
        }

    def _map_target_to_activity(self, target: str, session: Dict) -> Optional[str]:
        """
        Map a target reference (e.g., 'first', 'second', '3', 'Mood Meter') to an activity ID.
        Returns activity ID like 'A1', 'A2', etc., or None if not found.
        """
        
        last_results = session.get("last_results", [])
        
        if not last_results:
            return None
        
        target_lower = target.lower()
        
        # Handle ordinal references
        ordinal_map = {
            "first": 0,
            "1st": 0,
            "second": 1,
            "2nd": 1,
            "third": 2,
            "3rd": 2,
            "fourth": 3,
            "4th": 3,
            "fifth": 4,
            "5th": 4,
            "sixth": 5,
            "6th": 5,
            "seventh": 6,
            "7th": 6,
            "eighth": 7,
            "8th": 7,
        }
        
        # Check ordinals
        for ordinal, index in ordinal_map.items():
            if ordinal in target_lower:
                if index < len(last_results):
                    return f"A{index + 1}"
                else:
                    print(f"⚠️ [MAP] Ordinal '{ordinal}' out of bounds (have {len(last_results)} results)")
                    return None
        
        # Check for numeric references
        import re
        numbers = re.findall(r'\d+', target)
        if numbers:
            num = int(numbers[0])
            if 1 <= num <= len(last_results):
                return f"A{num}"
            else:
                print(f"⚠️ [MAP] Number {num} out of bounds (have {len(last_results)} results)")
                return None
        
        # Check for activity name match
        for idx, activity in enumerate(last_results, 1):
            title = activity.get("title", "").lower()
            if target_lower in title or title in target_lower:
                return f"A{idx}"
        
        print(f"⚠️ [MAP] Could not resolve target: {target}")
        return None


# Global instance
db_only_conversation_engine = DBOnlyConversationEngine()