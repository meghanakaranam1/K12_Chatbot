"""
Targeted modification system - intelligently modify specific parts of activities
Uses LLM-based classification with zero hardcoded patterns.
"""

import re
import json
from typing import Dict, Optional, Tuple, Any

class TargetedModifier:
    """
    Intelligently modifies specific parts of an activity instead of regenerating everything.
    Uses LLM to classify modification type and extract parameters - zero hardcoded patterns.
    """
    
    def __init__(self):
        pass
    
    def detect_modification_target(self, instruction: str) -> Tuple[str, Dict[str, Any]]:
        """
        Use LLM to detect what part of the activity needs modification.
        
        Returns:
            (target_type, target_info)
        """
        from services.ai_services import openai_service
        
        prompt = f"""Analyze this modification request and extract all relevant information.

Modification request: "{instruction}"

Return a JSON object with:
{{
    "type": "step|reflection|materials|examples|time|grade|objective|modifications|full",
    "target": {{
        "step_number": <int or null>,
        "count": <int or null>,
        "value": <string or null>,
        "action": <string describing what to do, or null>,
        "content": <string describing what content is affected, or null>
    }}
}}

Field explanations:
- "type": The SECTION of the activity being modified (step, reflection, materials, etc.)
- "step_number": Which step number (1, 2, 3...) if modifying a step
- "count": How many items to add/modify (for reflection questions, materials, etc.)
- "value": The new value (for time changes, grade levels, etc.)
- "action": Natural language description of what to do (e.g., "add examples", "make more detailed", "change wording", etc.)
- "content": What is being modified (e.g., "examples", "sub-steps", "wording", etc.)

Examples:
- "add examples to step 1" → {{"type": "step", "target": {{"step_number": 1, "action": "add examples", "content": "examples"}}}}
- "make step 2 more detailed" → {{"type": "step", "target": {{"step_number": 2, "action": "make more detailed", "content": "all content"}}}}
- "add 2 more reflection questions" → {{"type": "reflection", "target": {{"count": 2, "action": "add"}}}}
- "make it 20 minutes" → {{"type": "time", "target": {{"value": "20 min", "action": "change duration"}}}}
- "rephrase step 3 for clarity" → {{"type": "step", "target": {{"step_number": 3, "action": "rephrase for clarity", "content": "all content"}}}}
- "add visual aids to the materials" → {{"type": "materials", "target": {{"action": "add visual aids", "content": "visual aids"}}}}

Return ONLY valid JSON, no explanation."""

        try:
            response = openai_service.chat(
                messages=[
                    {"role": "system", "content": "You extract modification parameters from natural language. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0
            )
            
            result = json.loads(response)
            mod_type = result.get('type', 'full')
            target_info = result.get('target', {})
            
            print(f"🎯 [LLM CLASSIFY] Type: {mod_type}, Target: {target_info}")
            
            return mod_type, target_info
            
        except Exception as e:
            print(f"⚠️ [LLM CLASSIFY] Error: {e}, falling back to full modification")
            return 'full', {}
    
    def extract_step(self, directions: str, step_num: int) -> Optional[str]:
        """Extract a specific step from directions using flexible pattern matching."""
        print(f"🔍 [EXTRACT] Looking for step {step_num} in directions...")
        
        if not directions:
            print(f"⚠️ [EXTRACT] No directions provided")
            return None
        
        print(f"🔍 [EXTRACT] Directions preview: {directions[:200]}")
        
        # Try multiple patterns (from most specific to least)
        patterns = [
            rf'^## {step_num}\. \*{{0,2}}(.+?)\*{{0,2}}\s*$',  # ## 1. **Title**
            rf'^## {step_num}\. (.+)$',                         # ## 1. Title
            rf'^{step_num}\. \*{{0,2}}(.+?)\*{{0,2}}\s*$',     # 1. **Title**
            rf'^{step_num}\. (.+)$',                            # 1. Title
        ]
        
        step_match = None
        for pattern in patterns:
            step_match = re.search(pattern, directions, re.MULTILINE)
            if step_match:
                print(f"✅ [EXTRACT] Found step {step_num} with pattern: {pattern}")
                break
        
        if not step_match:
            print(f"⚠️ [EXTRACT] Could not find step {step_num}")
            return None
        
        step_start = step_match.start()
        
        # Find where step ends
        next_step_num = step_num + 1
        next_patterns = [
            rf'^## {next_step_num}\.',
            rf'^{next_step_num}\.',
        ]
        
        step_end = len(directions)
        for pattern in next_patterns:
            next_match = re.search(pattern, directions[step_start + 1:], re.MULTILINE)
            if next_match:
                step_end = step_start + next_match.start() + 1
                print(f"🔍 [EXTRACT] Step ends at next step")
                break
        
        # If no next step, look for next section
        if step_end == len(directions):
            section_match = re.search(r'\n## \*{0,2}[A-Z]', directions[step_start + 1:])
            if section_match:
                step_end = step_start + section_match.start() + 1
                print(f"🔍 [EXTRACT] Step ends at next section")
        
        extracted = directions[step_start:step_end].strip()
        print(f"✅ [EXTRACT] Extracted {len(extracted)} chars")
        return extracted

    def replace_step(self, directions: str, step_num: int, new_step_content: str) -> str:
        """Replace a specific step in directions with new content."""
        if not directions:
            return new_step_content
        
        # Find the step using flexible patterns
        patterns = [
            (rf'^(##\s+)?{step_num}\.\s+', re.MULTILINE),
        ]
        
        step_match = None
        for pattern, flags in patterns:
            step_match = re.search(pattern, directions, flags)
            if step_match:
                break
        
        if not step_match:
            # Step not found - append at end
            print(f"⚠️ [REPLACE] Step {step_num} not found, appending")
            return directions + "\n\n" + new_step_content
        
        step_start = step_match.start()
        
        # Find where this step ends
        next_step_num = step_num + 1
        next_patterns = [
            (rf'^(##\s+)?{next_step_num}\.', re.MULTILINE),
        ]
        
        step_end = None
        for pattern, flags in next_patterns:
            next_match = re.search(pattern, directions[step_start + 1:], flags)
            if next_match:
                step_end = step_start + next_match.start() + 1
                break
        
        if step_end is None:
            # Look for next section
            section_match = re.search(r'\n##\s+\*{0,2}[A-Z]', directions[step_start + 1:])
            if section_match:
                step_end = step_start + section_match.start() + 1
            else:
                step_end = len(directions)
        
        # Replace the step
        return directions[:step_start] + new_step_content + "\n\n" + directions[step_end:]
    
    def modify_step(self, step_content: str, instruction: str, action: Optional[str], content: Optional[str], source_content: Optional[str] = None) -> str:
        """Use LLM to modify a specific step."""
        from services.ai_services import openai_service
        
        source_context = ""
        if source_content and len(source_content) > 100:
            source_context = f"\n\n### Source Material (for reference):\n{source_content[:1500]}...\n"
        
        # Build action guidance from the natural language action
        action_guidance = f"Action to perform: {action}" if action else "Modify as requested in the instruction"
        
        prompt = f"""You are modifying ONE STEP of an activity.

### Current Step Content:
{step_content}

{source_context}

### Modification Request:
{instruction}

### {action_guidance}

### Rules:
1. Keep the step number and title format (e.g., "## 1. Title" or "1. Title")
2. Keep the same bullet format for sub-steps
3. Follow the modification request precisely
4. Return the COMPLETE modified step (not just what you added)
5. Use bullets (-) for sub-steps, not letters (a. b. c.)
6. Be specific and detailed

Return ONLY the modified step, nothing else."""

        try:
            modified_step = openai_service.chat(
                messages=[
                    {"role": "system", "content": "You modify ONE step of an activity while preserving formatting."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            return modified_step.strip()
        except Exception as e:
            print(f"❌ [MODIFY STEP] Error: {e}")
            return step_content
    
    def add_reflection_questions(self, current_reflection: str, count: int, source_content: Optional[str] = None) -> str:
        """Add reflection questions to existing ones."""
        from services.ai_services import openai_service
        
        source_context = ""
        if source_content and len(source_content) > 100:
            source_context = f"\n\n### Source Material:\n{source_content[:1500]}...\n"
        
        prompt = f"""Add {count} reflection questions to this activity.

### Current Reflection Questions:
{current_reflection if current_reflection else "(none yet)"}

{source_context}

### Rules:
1. Add EXACTLY {count} new questions
2. Use bullet format (-)
3. Make questions thoughtful and age-appropriate
4. Don't duplicate existing questions
5. Return ALL reflection questions (existing + new)

Return ONLY the bullet list of questions, nothing else."""

        try:
            updated_reflection = openai_service.chat(
                messages=[
                    {"role": "system", "content": f"You add exactly {count} reflection questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return updated_reflection.strip()
        except Exception as e:
            print(f"❌ [ADD REFLECTION] Error: {e}")
            return current_reflection
    
    def modify_activity(
        self,
        activity: Dict[str, str],
        instruction: str,
        source_content: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """
        Main entry point - intelligently modify the activity based on instruction.
        
        Returns:
            Modified activity dict, or None to signal fallback to full modification
        """
        # Use LLM to detect what needs modification
        target_type, target_info = self.detect_modification_target(instruction)
        
        print(f"🎯 [TARGETED] Modification type: {target_type}, target: {target_info}")
        
        # Handle based on type
        if target_type == "step":
            step_num = target_info.get('step_number')
            if not step_num:
                print(f"⚠️ [TARGETED] No step number extracted, falling back")
                return None
            
            # ✅ Check both "Directions" and "Steps" keys
            directions = activity.get("Directions") or activity.get("Steps") or ""
            
            # ✅ Debug what we actually got
            print(f"🔍 [TARGETED] Activity keys: {list(activity.keys())}")
            print(f"🔍 [TARGETED] Directions value: {type(directions)}, length: {len(str(directions))}")
            
            if not directions or not str(directions).strip():
                print(f"⚠️ [TARGETED] No directions found, falling back")
                return None
            
            step_content = self.extract_step(directions, step_num)
            if not step_content:
                print(f"⚠️ [TARGETED] Could not extract step {step_num}, falling back")
                return None
            
            # Modify the step
            action = target_info.get('action', 'modify')
            content = target_info.get('content')
            modified_step = self.modify_step(step_content, instruction, action, content, source_content)
            
            # Replace in directions
            new_directions = self.replace_step(directions, step_num, modified_step)
            
            # Update activity
            modified = dict(activity)
            modified["Directions"] = new_directions
            modified["Steps"] = new_directions
            
            print(f"✅ [TARGETED] Successfully modified step {step_num}")
            return modified
        
        elif target_type == "reflection":
            count = target_info.get('count', 2)
            current_reflection = activity.get("Reflection Questions", "")
            updated_reflection = self.add_reflection_questions(current_reflection, count, source_content)
            
            modified = dict(activity)
            modified["Reflection Questions"] = updated_reflection
            
            print(f"✅ [TARGETED] Added {count} reflection questions")
            return modified
        
        else:
            # Complex modification - use full modification path
            print(f"🔄 [TARGETED] Type '{target_type}' requires full modification")
            return None


# Global instance
targeted_modifier = TargetedModifier()