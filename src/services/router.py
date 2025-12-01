"""
Intent router with local greeting detection + LLM fallback
"""
import json
import re
from typing import Dict, List, Any, Optional


ROUTER_SYS = """You are an intent classifier for a classroom activity assistant.

Your ONLY job is to classify user intent into these categories:

1. capabilities - User wants to know what the system can do
2. db_search - User wants to find new activities
3. open_activity - User wants to open a specific activity
4. followup_customize - User wants to modify the current activity
5. followup_question - User has a question about the current activity
6. teaching_question - User wants teaching advice
7. acknowledgment - User is expressing thanks/satisfaction

Output ONLY valid JSON with: intent, target, confidence, reasoning."""


def _clean(s: str) -> str:
    """Clean and lowercase input"""
    return (s or "").strip().lower()


def _local_route(user_input: str, last_results, active_focus) -> Optional[Dict[str, Any]]:
    """
    Local detection for simple greetings AND activity name matching.
    """
    t = _clean(user_input)
    
    print(f"🔍 [LOCAL_ROUTE] Checking: '{t}'")
    
    # ============================================================================
    # 1. Handle single-word greetings locally
    # ============================================================================
    if t in ['hi', 'hello', 'hey', 'help', 'greetings', 'howdy', 'sup']:
        print(f"🎯 [LOCAL] Single-word greeting: '{t}' → capabilities")
        return {"intent": "capabilities", "target": None, "confidence": 0.95}
    
    # ============================================================================
    # 1.5. Check for compound requests FIRST - pass to LLM for multi-intent detection
    # ============================================================================
    compound_indicators = [' and also ', ' and then ', ' then ', ' and open ', ' and make ', ' and add ', ' and modify ', ' and change ']
    if any(indicator in t for indicator in compound_indicators):
        print(f"🔍 [LOCAL] Compound request detected (contains 'and also/then/etc') → passing to LLM for multi-intent analysis")
        return None  # Let LLM handle it
    
    # ============================================================================
    # 1.6. "Find X" should ALWAYS be search, not open (BUT only if not compound)
    # ============================================================================
    if t.startswith("find "):
        print(f"🎯 [LOCAL] 'find' pattern detected → db_search")
        return {"intent": "db_search", "target": None, "confidence": 0.95, "source": "local_find_pattern"}
    
    # ============================================================================
    # 2. Check if user is asking to open/show/expand an activity from last_results
    # Patterns: "show me X", "open X", "expand X", "I want see X"
    # ============================================================================
    if last_results and len(last_results) > 0:
        # Check for show/open/expand/view patterns
        show_pattern = r'^(show|open|expand|display|view|i want see|let me see|show me|open up)\s+(.+)$'
        match = re.match(show_pattern, t)
        
        if match:
            activity_query = match.group(2).strip()
            print(f"🔍 [LOCAL_ROUTE] Detected open request for: '{activity_query}'")
            
            # Special handling for pronouns: "it", "this", "that" → open first result
            if activity_query in ['it', 'this', 'that']:
                if last_results and len(last_results) > 0:
                    first_activity = last_results[0]
                    print(f"🎯 [LOCAL] Pronoun '{activity_query}' → opening first result: {first_activity.get('title')}")
                    return {
                        "intent": "open_activity",
                        "target": first_activity.get('display_id', 'A1'),
                        "row_index": first_activity.get('row_index'),
                        "title": first_activity.get('title'),
                        "payload": first_activity,
                        "confidence": 0.95,
                        "source": "local_pronoun_first_result"
                    }
            
            # Try to match against activity titles in last_results (for specific names)
            for i, activity in enumerate(last_results):
                title = activity.get('title', '').lower()
                
                # Fuzzy match: check if query is in title or title is in query
                # BUT: avoid matching short words like "it" inside other words
                if len(activity_query) >= 3:  # Only fuzzy match for 3+ char queries
                    if activity_query in title or title in activity_query:
                        print(f"🎯 [LOCAL] Matched '{activity_query}' to activity '{title}' → open_activity")
                        return {
                            "intent": "open_activity",
                            "target": activity.get('display_id', f'A{i+1}'),
                            "row_index": activity.get('row_index'),
                            "title": activity.get('title'),
                            "payload": activity,
                            "confidence": 0.95,
                            "source": "local_activity_match"
                        }
            
            print(f"🔍 [LOCAL_ROUTE] No match found for '{activity_query}' in last_results")
    
    # Everything else goes to LLM
    print(f"🔍 [LOCAL_ROUTE] Not a simple greeting or activity match, passing to LLM")
    return None


def route_intent_and_target(
    user_input: str,
    conversation_history: Optional[List[Dict]] = None,
    last_results: Optional[List[Dict]] = None,
    all_activities_seen: Optional[Dict] = None,
    active_focus: Optional[Dict] = None,
    session_data: Optional[Dict] = None,
    openai_service = None,
) -> Dict[str, Any]:
    """
    Route user intent with local detection first, then LLM fallback.
    
    Strategy:
      1) Try local routing (greetings only)
      2) Try LLM routing
      3) Default to db_search on error
    """
    print(f"🎯 [ROUTER] Processing: {user_input}")
    
    # Use session_data if provided, otherwise create empty dict
    if session_data is None:
        session_data = {}
    
    # Import openai_service if not provided
    if openai_service is None:
        from services.ai_services import openai_service as _openai_service
        openai_service = _openai_service
    
    # ====================================================================
    # STEP 1: Local detection (greetings only)
    # ====================================================================
    local = _local_route(user_input, last_results, active_focus)
    if local:
        print(f"✅ [LOCAL ROUTE] {local}")
        return local
    
    # ====================================================================
    # STEP 2: LLM routing
    # ====================================================================
    recent_titles = [r.get('title', '') for r in (last_results or [])[:5]]
    active_title = active_focus.get('title') if active_focus else None
    
    # ========== ADD CONVERSATION CONTEXT ==========
    last_bot_message = session_data.get('last_bot_message', '')
    last_opened = session_data.get('last_opened_index')
    
    conversation_context = ""
    if last_bot_message:
        conversation_context = f"""

🗨️ CONVERSATION CONTEXT:
Previous bot message: "{last_bot_message[:300]}"

🚨 CRITICAL: Context-aware routing rules:
1. If bot just suggested opening an activity and user responds with affirmative words:
   - "yes", "sure", "okay", "sounds good", "go ahead", "let's do it", "open it", "show me"
   → intent is "open_activity" (user accepting suggestion)
   
2. If bot showed search results and user responds:
   - "yes" → "open_activity" (opening suggested/first result)
   - "open 1", "show me 2" → "open_activity" (numbered selection)
   - "find something else" → "db_search" (new search)
"""
    # =============================================
    
    active_context = ""
    if active_title:
        active_context = f"""

IMPORTANT CONTEXT:
- Currently active activity: "{active_title}"
- If user says "this one", "use this", "this works" → they're referring to the active activity
- These should be "acknowledgment" intent
"""
    
    results_context = ""
    if recent_titles:
        results_context = f"""
🎯 IMPORTANT: Recent search results available. User can reference these activities:

{chr(10).join(f"{i+1}. {t}" for i, t in enumerate(recent_titles))}

🚨 CRITICAL RULES FOR ACTIVITY REFERENCES:
1. If user mentions ANY activity name above (exact or partial match) → "open_activity"
2. If user says position ("first one", "number 1", "A1") → "open_activity"
3. If user just says the activity name alone → "open_activity"

Examples that MUST be "open_activity":
- "show me [activity name]" → open_activity
- "open [activity name]" → open_activity
- "expand [activity name]" → open_activity
- "tell me about [activity name]" → open_activity
- "[activity name]" (just the name) → open_activity
- "the first one" / "number 1" → open_activity

Example that is "db_search":
- "find activities about [NEW topic not in list above]"
"""
    
    prompt = f"""You are routing user requests in an educational activity chatbot.
{conversation_context}
{active_context}
{results_context}

Current user message: "{user_input}"

Your task: Determine the user's intent and target.

Available intents:
- "capabilities": User wants to know what the system can do (generic question with NO specific topic)
- "db_search": User wants to find/search for NEW activities (has specific topic, context, or qualifiers)
- "open_activity": User wants to open/view a specific activity BY NAME (matches a title from recent results)
- "clarification_needed": User referenced MULTIPLE activities but system can only handle ONE at a time
- "followup_customize": User wants to modify the current activity
- "followup_question": User has a question about the current activity
- "teaching_question": User asking general teaching advice
- "acknowledgment": User is thanking you or expressing satisfaction
- "navigate": User wants to see more results
- "lost_context": User wants to modify but NO activity is currently open
- "multi_intent": User has MULTIPLE distinct intents in one message that should be executed sequentially

🚨 CRITICAL: If user mentions an activity name from the recent results list above, choose "open_activity" NOT "db_search"

Respond with JSON:
{{
  "intent": "one of the intents above",
  "target": "activity name if opening specific one, else null",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation",
  "clarification_message": "only if intent is clarification_needed, provide a helpful message",
  "intent_sequence": "only if intent is multi_intent, provide ordered list of intents"
}}

🚨 COMPOUND REQUEST DETECTION (CRITICAL):
If the user's message contains MULTIPLE DISTINCT INTENTS that should be executed in sequence, return "multi_intent".

A compound request has multiple actions connected by words like "and", "then", "also":
- "Find X and open Y and modify Z" → 3 intents: db_search, open_activity, followup_customize
- "Search for activities and open the first one" → 2 intents: db_search, open_activity
- "Open activity 2 and make it shorter" → 2 intents: open_activity, followup_customize

When you detect a compound request:
1. Set intent to "multi_intent"
2. Set confidence to 0.90
3. In intent_sequence, provide an ordered array of intent objects to execute:
   [
     {{"intent": "db_search", "target": null, "reasoning": "search for activities"}},
     {{"intent": "open_activity", "target": "second one", "reasoning": "open the second result"}},
     {{"intent": "followup_customize", "target": null, "reasoning": "make it shorter"}}
   ]

Examples of COMPOUND requests (multi_intent):
- "Find activities and also open the second one and make it shorter"
  → multi_intent with sequence: [db_search, open_activity, followup_customize]
  
- "Search for empathy activities and open the first one"
  → multi_intent with sequence: [db_search, open_activity]
  
- "Open activity 3 and add questions to it"
  → multi_intent with sequence: [open_activity, followup_customize]

Examples of SINGLE intent requests (NOT multi_intent):
- "Find 15 minute activities" → db_search (only one action)
- "Open the second one" → open_activity (only one action)
- "Make it shorter" → followup_customize (only one action)
- "Open 1 and 3 and 5" → clarification_needed (multiple targets, not multiple intents)

Key distinction:
- Multiple INTENTS (different actions) → multi_intent
- Multiple TARGETS (same action) → clarification_needed
- Single intent → return that intent directly



CRITICAL RULES FOR CAPABILITIES vs DB_SEARCH:

capabilities examples (asking what the system can do - NO specific topic):
- "what can you do"
- "what do you do"
- "how can you help me"
- "what is this"
- "what activities do you have" (ONLY if no topic/context follows)

db_search examples (asking for SPECIFIC activities - has topic/context):
- "what activities do you have for conflict resolution" → db_search (has topic: conflict)
- "what activities do you have for welcoming new students" → db_search (has topic: welcoming)
- "I'm teaching 6th grade and need activities for X" → db_search (has context)
- "show me activities about Y" → db_search (has topic: Y)
- "find activities for Z" → db_search (has topic: Z)
- "what activities do you have that help with X" → db_search (has qualifier)

KEY DISTINCTION:
- If the question includes a SPECIFIC TOPIC, GRADE, CONTEXT, or QUALIFIER after "what activities" → "db_search"
- If the question is truly GENERIC with no context (just "what activities do you have" and nothing else) → "capabilities"

OTHER RULES:
1. If user says "this one", "use this", "I'll use this" and there's an active activity → "acknowledgment"
2. If user says "first one", "number 2" and there are search results → "open_activity"
3. If user wants to change/modify current activity → "followup_customize"
4. If user asks how/what/why about current activity → "followup_question"
5. When in doubt between capabilities and db_search, choose "db_search" (better to search than show capabilities)
"""

    try:
        response = openai_service.chat(
            messages=[
                {"role": "system", "content": ROUTER_SYS},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.1
        )
        
        result = json.loads(response)
        result['source'] = 'llm_context'
        
        # Add metadata
        if active_title:
            result.setdefault('metadata', {})['active_title'] = active_title
        
        # Handle open_activity intent - extract activity details
        if result.get('intent') == 'open_activity' and last_results:
            target = result.get('target')
            if target:
                # Try to match target to an activity in last_results
                target_lower = target.lower()
                for i, activity in enumerate(last_results):
                    title = activity.get('title', '').lower()
                    # Match by position ("first", "1", "A1") or by title
                    if (target_lower in ['first', '1', 'a1'] and i == 0) or \
                       (target_lower in title or title in target_lower):
                        result['target'] = activity.get('display_id', f'A{i+1}')
                        result['row_index'] = activity.get('row_index')
                        result['title'] = activity.get('title')
                        result['payload'] = activity
                        break
        
        print(f"🎯 [ROUTER] LLM routing: intent={result.get('intent')}, target={result.get('target')}, confidence={result.get('confidence', 0):.2f}")
        print(f"🎯 [ROUTER] Reasoning: {result.get('reasoning', 'N/A')}")
        
        return result
    
    except Exception as e:
        print(f"⚠️  [ROUTER] LLM routing failed: {e}, defaulting to db_search")
        return {
            'intent': 'db_search',
            'confidence': 0.50,
            'source': 'error_fallback',
            'target': None,
            'reasoning': f'LLM routing error: {str(e)}'
        }
