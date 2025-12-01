"""
Smart Activity Modifier with Validation Loops
==============================================

This module handles activity modifications using:
1. Structured outputs for numerical targets
2. Validation loops with retry logic
3. Programmatic fallbacks when LLM fails

Instead of relying on LLM to follow complex instructions perfectly,
we VALIDATE outputs and retry until they meet requirements.
"""

import json
import re
import html
from typing import Dict, Any, Optional, Tuple, List
from services.ai_services import openai_service


class SmartModifier:
    """Handles activity modifications with validation and retry logic"""
    
    def __init__(self):
        self.max_retries = 2  # Don't retry forever
    
    def _validate_request(
        self,
        user_request: str,
        activity: Dict[str, Any],
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to validate if the modification request is valid, contradictory, or impossible.
        Returns: {"status": "valid"|"contradictory"|"impossible", "reason": str, "clarification": str}
        """
        
        # Summarize current activity state
        activity_summary = {
            "title": activity.get("Activity Name", "Unknown"),
            "current_time": activity.get("Time", "Unknown"),
            "reflection_questions_count": len([
                q for q in activity.get("Reflection Questions", "").split('\n') 
                if q.strip().startswith('-')
            ]),
            "directions_length": len(activity.get("Directions", ""))
        }
        
        # Build validation prompt
        validation_prompt = f"""You are analyzing a user's modification request for potential issues.

**Current Activity State:**
- Title: {activity_summary['title']}
- Current Duration: {activity_summary['current_time']}
- Number of Reflection Questions: {activity_summary['reflection_questions_count']}
- Directions Length: {activity_summary['directions_length']} characters

**User's Modification Request:**
"{user_request}"

**Your Task:**
Analyze if this request is valid, contradictory, or impossible to fulfill.

**Examples:**

CONTRADICTORY REQUEST:
- Request: "Make it 30 minutes but keep it under 10 minutes"
- Response: {{"status": "contradictory", "reason": "Cannot be both 30 minutes and under 10 minutes", "clarification": "I notice you've asked to make it 30 minutes AND keep it under 10 minutes. Which would you prefer: 30 minutes or under 10 minutes?"}}

IMPOSSIBLE REQUEST:
- Request: "Remove the first 10 reflection questions" (when only 3 exist)
- Response: {{"status": "impossible", "reason": "Only 3 reflection questions exist", "clarification": "This activity only has 3 reflection questions. I can't remove 10 questions. Would you like me to remove all 3 questions?"}}

VALID REQUEST:
- Request: "Make it shorter"
- Response: {{"status": "valid", "reason": "", "clarification": ""}}

**CRITICAL RULES:**
1. If status is "contradictory" or "impossible", you MUST provide a helpful clarification question
2. The clarification should be a polite, specific question to resolve the issue
3. Your response must be ONLY valid JSON, no other text
4. DO NOT include markdown code blocks (```json) - just the JSON object

**Respond with this exact JSON structure:**
{{
  "status": "valid" | "contradictory" | "impossible",
  "reason": "Brief explanation (empty if valid)",
  "clarification": "Specific question to ask user (empty if valid, REQUIRED if contradictory/impossible)"
}}
"""
        
        # Call LLM
        messages = [{"role": "user", "content": validation_prompt}]
        
        print(f"🔍 [VALIDATE] Checking request for contradictions/impossibilities...")
        
        try:
            response = openai_service.chat(
                messages=messages,
                model="gpt-4o-mini",
                temperature=0.1  # Low temperature for consistent validation
            )
            
            # Parse JSON response
            response_text = response.strip()
            
            # Strip markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.replace("```json\n", "").replace("```\n", "").replace("```", "").strip()
            
            validation_result = json.loads(response_text)
            
            print(f"✅ [VALIDATE] Status: {validation_result.get('status')}")
            if validation_result.get('status') != 'valid':
                print(f"⚠️ [VALIDATE] Reason: {validation_result.get('reason')}")
                print(f"⚠️ [VALIDATE] Clarification: {validation_result.get('clarification')}")
            
            return validation_result
            
        except Exception as e:
            print(f"❌ [VALIDATE] Error parsing validation response: {e}")
            # If validation fails, proceed anyway (fail-open)
            return {"status": "valid", "reason": "", "clarification": ""}
    
    def modify_activity(
        self,
        activity: Dict[str, Any],
        user_request: str,
        source_content: str = "",
        conversation_history: List[Dict] = None,
        original_activity: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Modify an activity based on user request using smart LLM-powered handlers
        
        Args:
            activity: Current activity to modify
            user_request: User's modification request
            source_content: Original source content for context
            conversation_history: List of previous conversation messages
            original_activity: The unmodified original activity (before any changes)
        """
        if conversation_history is None:
            conversation_history = []
        
        if original_activity is None:
            original_activity = activity  # Fallback if not provided
        
        # ✅ STEP 1: Validate the request first
        validation = self._validate_request(user_request, activity, conversation_history)
        
        # ✅ STEP 2: If not valid, return clarification request
        if validation.get("status") != "valid":
            return {
                "needs_clarification": True,
                "clarification_message": validation.get("clarification"),
                "validation_reason": validation.get("reason"),
                "validation_status": validation.get("status")
            }
        
        # ✅ STEP 3: If valid, proceed with modification
        # Detect what type of modification this is
        mod_type = self._detect_modification_type(user_request)
        
        if mod_type == "shorten":
            return self._handle_shortening(activity, user_request, source_content, conversation_history, original_activity)
        elif mod_type == "modify_questions":
            # Single LLM-powered handler for ALL question operations
            return self._handle_modify_questions(activity, user_request, source_content, conversation_history, original_activity)
        else:
            # Generic modification - use standard LLM
            return self._handle_generic(activity, user_request, source_content, conversation_history, original_activity)
    
    def _detect_modification_type(self, user_request: str) -> str:
        """Use LLM to intelligently detect modification type - no hardcoded keywords"""
        
        prompt = f"""Classify this modification request into ONE category:

Request: "{user_request}"

Categories:
- "shorten_directions" - if asking to shorten/condense/reduce the directions/steps
- "modify_questions" - if asking to add/remove/change/replace reflection questions
- "generic" - anything else (change time, adapt grade level, add examples, etc.)

Return ONLY the category name, nothing else."""

        response = openai_service.chat(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            temperature=0.1
        ).strip().lower()
        
        # Normalize response
        if "shorten" in response:
            return "shorten"
        elif "question" in response:
            return "modify_questions"
        else:
            return "generic"
    
    def _handle_shortening(
        self,
        activity: Dict[str, Any],
        user_request: str,
        source_content: str,
        conversation_history: List[Dict] = None,
        original_activity: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle shortening with VALIDATION LOOP
        
        This is the key insight: Don't trust LLM to follow instructions.
        VERIFY output and retry if needed.
        """
        if conversation_history is None:
            conversation_history = []
        
        original_directions = activity.get("Directions", "")
        original_length = len(original_directions)
        target_length = int(original_length * 0.45)  # 45% of original
        
        print(f"🎯 [SHORTEN] Original: {original_length} chars, Target: {target_length} chars")
        
        # Try up to max_retries times
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                print(f"🔄 [RETRY] Attempt {attempt + 1}/{self.max_retries + 1}")
            
            # Use structured output to force LLM to think about numbers
            prompt = self._build_shortening_prompt(
                activity,
                original_length,
                target_length,
                attempt_number=attempt,
                conversation_history=None  # Don't add to prompt, we'll use messages
            )
            
            # ✅ FIX: Build messages array with actual conversation history
            messages = []
            
            # Add last 3 conversation turns (6 messages total)
            if conversation_history:
                for msg in conversation_history[-6:]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if content and role in ["user", "assistant"]:
                        messages.append({
                            "role": role,
                            "content": content[:500]  # Truncate to 500 chars
                        })
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            print(f"🔍 [SHORTEN] Passing {len(messages)} messages to LLM (including {len(conversation_history[-6:]) if conversation_history else 0} history messages)")
            
            response = openai_service.chat(
                messages=messages,  # FIX: Pass full conversation!
                model="gpt-4o-mini",
                temperature=0.3  # Lower temp for more consistent output
            )
            
            # Parse response
            try:
                # Try to extract JSON if present
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    shortened = data.get("shortened_directions", "")
                else:
                    # Fallback: just use the text
                    shortened = response
                
                shortened_length = len(shortened)
                
                # VALIDATION: Did it actually shorten?
                if shortened_length <= target_length * 1.3:  # Allow 30% margin
                    print(f"✅ [SHORTEN] Success! {shortened_length} chars ({shortened_length/original_length*100:.1f}% of original)")
                    activity["Directions"] = shortened
                    return activity
                else:
                    print(f"❌ [SHORTEN] Failed: {shortened_length} chars (still {shortened_length/original_length*100:.1f}% of original)")
                    
            except Exception as e:
                print(f"⚠️  [SHORTEN] Error parsing response: {e}")
        
        # If all retries failed, use PROGRAMMATIC fallback
        print(f"⚠️  [FALLBACK] LLM couldn't shorten reliably, using programmatic truncation")
        shortened = self._programmatic_shorten(original_directions, target_length)
        activity["Directions"] = shortened
        return activity
    
    def _build_shortening_prompt(
        self,
        activity: Dict[str, Any],
        original_length: int,
        target_length: int,
        attempt_number: int = 0,
        conversation_history: List[Dict] = None
    ) -> str:
        """Build prompt for shortening with clear numerical targets"""
        # ✅ FIX: Don't format history here - it's passed as messages now
        
        directions = activity.get("Directions", "")
        
        # Make prompt MORE aggressive on retries
        if attempt_number == 0:
            aggression = "Combine related steps and remove examples"
        else:
            aggression = "BE RUTHLESS. Merge ALL steps possible, remove ALL examples, use ULTRA-concise language"
        
        return f"""You are shortening activity directions.

**CRITICAL NUMBERS:**
- Original length: {original_length} characters
- Target length: {target_length} characters (45% of original)
- You MUST produce output that is {target_length} ± 200 characters

**ORIGINAL DIRECTIONS:**
{directions}

**YOUR TASK:**
{aggression}. Your output must be approximately {target_length} characters.

**OUTPUT FORMAT:**
Return ONLY the shortened directions as plain text. Do not include any JSON, explanations, or metadata.

**VERIFICATION:**
Before responding, count your output characters. If it's more than {target_length + 200}, go back and cut MORE.

Begin your shortened directions:"""
    
    def _programmatic_shorten(self, directions: str, target_length: int) -> str:
        """
        Programmatic fallback: Actually shorten the text if LLM fails
        
        This is your safety net - it's not as smart as LLM, but it WORKS.
        """
        # Split into steps
        steps = re.split(r'\n##\s+\d+\.', directions)
        
        if len(steps) <= 1:
            # Not properly formatted, just truncate smartly
            return self._smart_truncate(directions, target_length)
        
        # Keep first step (intro), merge middle steps, keep last step
        header = steps[0] if steps else ""
        remaining_steps = steps[1:] if len(steps) > 1 else []
        
        if len(remaining_steps) == 0:
            return self._smart_truncate(header, target_length)
        
        # Reconstruct with fewer steps
        shortened = header
        
        # Take every other step, or combine substeps
        for i, step in enumerate(remaining_steps[::2]):  # Skip every other
            # Remove examples (sentences with "For example", "For instance")
            step = re.sub(r'[^.]*\bfor (example|instance)[^.]*\.', '', step, flags=re.IGNORECASE)
            # Remove elaborations
            step = re.sub(r'[^.]*\b(this shows|make sure|remember that|note that)[^.]*\.', '', step, flags=re.IGNORECASE)
            shortened += f"\n## {i+1}. " + step.strip()
        
        return shortened[:target_length + 200]  # Hard cap
    
    def _smart_truncate(self, text: str, target_length: int) -> str:
        """Truncate text at sentence boundaries"""
        if len(text) <= target_length:
            return text
        
        # Find last sentence boundary before target
        truncated = text[:target_length]
        last_period = truncated.rfind('.')
        
        if last_period > target_length * 0.7:  # At least 70% of target
            return text[:last_period + 1]
        else:
            return truncated + "..."
    
    def _handle_modify_questions(
        self,
        activity: Dict[str, Any],
        user_request: str,
        source_content: str,
        conversation_history: List[Dict] = None,
        original_activity: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Smart LLM-powered handler for ALL question modifications.
        
        Handles: add, remove, replace, change any number of questions
        No hardcoded logic - let LLM understand the intent.
        """
        
        if conversation_history is None:
            conversation_history = []
        
        if original_activity is None:
            original_activity = activity
        
        current_questions = activity.get("Reflection Questions", "")
        original_questions = original_activity.get("Reflection Questions", "")
        
        print(f"🎯 [MODIFY_Q] Request: {user_request}")
        print(f"🎯 [MODIFY_Q] Current questions: {len(current_questions.split(chr(10)) if current_questions else [])} lines")
        
        # ✅ FIX: Don't add history to prompt text, use actual messages
        prompt = f"""You are modifying reflection questions for an educational activity.

**ORIGINAL REFLECTION QUESTIONS (before any modifications):**
{original_questions}

**CURRENT REFLECTION QUESTIONS (may have been modified):**
{current_questions}

**User Request:**
{user_request}

**Your Task:**
Carefully read the user's request and conversation history.
- If the user is asking to undo/remove/restore a previous change (e.g., "remove this", "undo that"), consider what "this" or "that" refers to based on the conversation.
- If they're referring to questions that were recently added, remove only those questions.
- If they're asking to restore to original, use the ORIGINAL questions above.
- Otherwise, modify the CURRENT questions according to their request.

**Important:**
- Preserve the bullet format: use "- " (dash space) for each question
- Return ALL questions (after modification)
- Number of questions should match what the user requested

**Output Format:**
Return ONLY the modified reflection questions, one per line, each starting with "- "
No explanations, no markdown, just the questions.

Begin your modified questions:"""

        # ✅ FIX: Build messages array with actual conversation history
        messages = []
        
        # Add last 3 conversation turns (6 messages total)
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content and role in ["user", "assistant"]:
                    messages.append({
                        "role": role,
                        "content": content[:500]  # Truncate to 500 chars
                    })
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        print(f"🔍 [MODIFY_Q] Passing {len(messages)} messages to LLM (including {len(conversation_history[-6:]) if conversation_history else 0} history messages)")
        
        response = openai_service.chat(
            messages=messages,  # FIX: Pass full conversation!
            model="gpt-4o-mini",
            temperature=0.3
        )
        
        # ✅ FIX: Escape HTML to prevent XSS
        response = html.escape(response)
        
        # Validate we got questions back
        new_questions = [q.strip() for q in response.split('\n') if q.strip() and q.strip().startswith('-')]
        
        if new_questions:
            activity["Reflection Questions"] = '\n'.join(new_questions)
            print(f"✅ [MODIFY_Q] Success! {len(new_questions)} questions")
        else:
            print(f"⚠️  [MODIFY_Q] LLM returned no valid questions, keeping original")
        
        return activity
    
    def _handle_add_questions(
        self,
        activity: Dict[str, Any],
        user_request: str,
        source_content: str
    ) -> Dict[str, Any]:
        """Handle adding reflection questions with validation"""
        
        # Extract how many to add
        numbers = re.findall(r'\d+', user_request)
        num_to_add = int(numbers[0]) if numbers else 3
        
        current_questions = activity.get("Reflection Questions", "")
        current_count = len([q for q in current_questions.split('\n') if q.strip()])
        
        print(f"🎯 [ADD_Q] Current: {current_count}, Adding: {num_to_add}")
        
        prompt = f"""Add {num_to_add} reflection questions to this activity.

**CURRENT REFLECTION QUESTIONS:**
{current_questions}

**ACTIVITY CONTEXT:**
{activity.get('Objective', '')}

**YOUR TASK:**
Add exactly {num_to_add} NEW reflection questions that:
1. Are different from existing questions
2. Relate to the activity objective
3. Are numbered sequentially after existing questions

**OUTPUT FORMAT:**
Return ALL reflection questions (existing + new) as a numbered list:
1. [existing question 1]
2. [existing question 2]
...
{current_count + 1}. [new question 1]
{current_count + 2}. [new question 2]
...

Begin your numbered list:"""
        
        response = openai_service.chat(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini"
        )
        
        # Validate: Count questions
        new_questions = [q for q in response.split('\n') if q.strip() and re.match(r'^\d+\.', q.strip())]
        
        if len(new_questions) == current_count + num_to_add:
            print(f"✅ [ADD_Q] Success! Added {num_to_add} questions")
            activity["Reflection Questions"] = response
        else:
            print(f"⚠️  [ADD_Q] Got {len(new_questions)} questions, expected {current_count + num_to_add}")
            activity["Reflection Questions"] = response
        
        return activity
    
    def _handle_remove_questions(
        self,
        activity: Dict[str, Any],
        user_request: str,
        source_content: str
    ) -> Dict[str, Any]:
        """Handle removing reflection questions programmatically"""
        
        # Extract how many to remove
        numbers = re.findall(r'\d+', user_request)
        num_to_remove = int(numbers[0]) if numbers else 1
        
        current_questions = activity.get("Reflection Questions", "")
        questions_list = [q.strip() for q in current_questions.split('\n') if q.strip() and (re.match(r'^\d+\.', q.strip()) or re.match(r'^-\s', q.strip()))]
        
        print(f"🎯 [REMOVE_Q] Current: {len(questions_list)}, Removing: {num_to_remove}")
        
        # PROGRAMMATIC: Just remove first N questions
        remaining = questions_list[num_to_remove:]
        
        # Renumber
        # Preserve bullet format for bullet questions
        renumbered = []
        for i, q in enumerate(remaining):
            if q.startswith('-'):
                # Keep as bullet
                renumbered.append(q)
            else:
                # Renumber if numbered
                parts = q.split('.', 1)
                if len(parts) > 1:
                    renumbered.append(f"{i+1}. {parts[1].strip()}")
                else:
                    renumbered.append(f"{i+1}. {q}")
        
        activity["Reflection Questions"] = '\n'.join(renumbered)
        print(f"✅ [REMOVE_Q] Removed {num_to_remove}, {len(remaining)} remaining")
        
        return activity
    
    def _handle_generic(
        self,
        activity: Dict[str, Any],
        user_request: str,
        source_content: str,
        conversation_history: List[Dict] = None,
        original_activity: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Handle generic modifications using existing document_processor logic"""
        if conversation_history is None:
            conversation_history = []
        
        if original_activity is None:
            original_activity = activity
        
        # ✅ FIX: Don't add history to prompt text, use actual messages
        # Build the full prompt
        prompt = f"""You are modifying an educational activity.

**ORIGINAL ACTIVITY (before any modifications):**
{json.dumps(original_activity, indent=2)}

**CURRENT ACTIVITY (may have been modified from original):**
{json.dumps(activity, indent=2)}

**USER REQUEST:**
{user_request}

**AVAILABLE SOURCE CONTENT:**
{source_content[:2000] if source_content else "No additional source content"}

**YOUR TASK:**
Analyze the user's request and conversation history carefully.

**CRITICAL RULES:**
1. If the user is asking to undo/remove/restore a previous modification (keywords: "remove this", "undo this", "restore", "go back", "revert"):
   - Look at the conversation history to understand what "this" refers to
   - If they want to undo a shortening or restoration, return the **ORIGINAL ACTIVITY EXACTLY** as shown above (first JSON object)
   - DO NOT modify, rephrase, or regenerate - copy it VERBATIM from the ORIGINAL ACTIVITY section
   
2. If the user wants to remove recently added items (e.g., "remove those questions I just added"):
   - Remove only those specific items from the CURRENT activity
   - Keep other modifications intact

3. If the user is requesting a new modification:
   - Apply it to the CURRENT ACTIVITY

**IMPORTANT:** When restoring to original, you must return the exact JSON from "ORIGINAL ACTIVITY (before any modifications)" section above - word-for-word, character-for-character. Do not paraphrase or rewrite.

Return ONLY the complete activity as a JSON object with these fields:
- Title
- Time
- Objective  
- Materials
- Directions
- Reflection Questions
- Modification Suggestions

Ensure all fields are present and properly formatted.

Return the activity as JSON:"""

        # ✅ FIX: Build messages array with actual conversation history
        messages = []
        
        # Add last 3 conversation turns (6 messages total)
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content and role in ["user", "assistant"]:
                    messages.append({
                        "role": role,
                        "content": content[:500]  # Truncate to 500 chars
                    })
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        print(f"🔍 [GENERIC] Passing {len(messages)} messages to LLM (including {len(conversation_history[-6:]) if conversation_history else 0} history messages)")
        
        # Debug: Print first 200 chars of Directions from both activities
        print(f"🔍 [DEBUG] ORIGINAL activity Directions (first 200 chars): {original_activity.get('Directions', '')[:200]}")
        print(f"🔍 [DEBUG] CURRENT activity Directions (first 200 chars): {activity.get('Directions', '')[:200]}")
        
        response = openai_service.chat(
            messages=messages,  # FIX: Pass full conversation!
            model="gpt-4o-mini",
            temperature=0.3
        )
        
        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                modified = json.loads(json_match.group())
                return modified
            else:
                # Fallback: return original with note
                print("⚠️  [GENERIC] Could not parse JSON, returning original")
                return activity
        except Exception as e:
            print(f"⚠️  [GENERIC] Error parsing response: {e}")
            return activity


# Global instance
smart_modifier = SmartModifier()