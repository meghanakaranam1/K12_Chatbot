# services/conversation_manager.py
"""
Conversation management with structured activities
"""
import re
from typing import Dict, Any, List
from services.activity_customizer import activity_customizer
from processing.document_processor import parse_legacy_activity, strip_added_headers, Activity
from services.ai_services import openai_service

class ConversationManager:
    """Manages conversation flow with structured activities"""
    
    def __init__(self):
        self.context: Dict[str, Any] = {
            "current_activity": None,  # store structured Activity here
        }
        # Cache for generated content to reduce token usage
        self._examples_cache = {}
        self._questions_cache = {}
        self._materials_cache = {}

    def _clean_md(self, md: str) -> str:
        """Strip markdown fences and extra whitespace for clean UI rendering."""
        if not md:
            return ""
        text = md.strip()
        # Remove code fences like ```markdown ... ```
        text = text.replace("```markdown", "").replace("```", "").strip()
        return text

    def set_current_activity(self, legacy_activity_dict: Dict[str, Any]):
        """Set the current activity from legacy format"""
        # Clean any existing "Added..." headers before parsing
        for key in legacy_activity_dict:
            if isinstance(legacy_activity_dict[key], str):
                legacy_activity_dict[key] = strip_added_headers(legacy_activity_dict[key])
        
        self.context["current_activity"] = parse_legacy_activity(legacy_activity_dict)

    def _parse_dynamic_count(self, user_text: str, default: int = 5) -> int:
        """Parse count from user input (e.g., 'add 3 examples' -> 3)"""
        m = re.search(r"\b(add|give|include|create)\s+(\d{1,2})\s+(?:example|examples|question|questions|material|materials)\b", user_text.lower())
        return int(m.group(2)) if m else default

    def handle_user_enhancement(self, user_text: str) -> Dict[str, Any]:
        """Handle user enhancement requests with structured updates"""
        act = self.context.get("current_activity")
        if not act:
            return {"type": "status", "message": "Please pick an activity first."}

        user_lower = user_text.lower()

        # Examples intent
        if re.search(r"(?:add|give|include|create).*(?:example|examples)", user_lower):
            return self._handle_examples_request(act, user_text)
        
        # Discussion questions intent
        if re.search(r"(?:add|give|include|create).*(?:discussion|question|questions)", user_lower):
            return self._handle_discussion_questions_request(act, user_text)
        
        # Materials intent
        if re.search(r"(?:add|give|include|create).*(?:material|materials)", user_lower):
            return self._handle_materials_request(act, user_text)
        
        # Directions intent
        if re.search(r"(?:add|give|include|create|replace|update).*(?:direction|directions|step|steps)", user_lower):
            return self._handle_directions_request(act, user_text)
        
        # Objectives intent
        if re.search(r"(?:add|give|include|create).*(?:objective|objectives|goal|goals)", user_lower):
            return self._handle_objectives_request(act, user_text)
        
        # Assessment intent
        if re.search(r"(?:add|give|include|create).*(?:assessment|evaluation|check)", user_lower):
            return self._handle_assessment_request(act, user_text)

        # Fallback
        return {"type": "status", "message": "Tell me what to add (examples, questions, materials, steps, objectives, assessment…)"}

    def _handle_examples_request(self, act: Activity, user_text: str) -> Dict[str, Any]:
        """Handle examples enhancement request using LLM generator"""
        # Determine if replace or merge
        replace = re.search(r"(?:replace|new|different|instead)", user_text.lower()) is not None
        
        # Parse dynamic count
        count = self._parse_dynamic_count(user_text, 5)
        
        # Get context for better tailoring
        grade = self.context.get("grade_level")
        subject = self.context.get("subject_area")
        
        # Check cache first
        cache_key = (act.title, grade or "", subject or "", count)
        if cache_key in self._examples_cache:
            examples = self._examples_cache[cache_key]
        else:
            # Generate examples using LLM
            prompt = f"""Generate {count} specific, practical examples for this classroom activity.

Activity: {act.title}
Grade Level: {grade or "not specified"}
Subject: {subject or "not specified"}

Existing examples: {act.sections.get("examples", [])}

Return {count} concrete examples that students can use. Format as a simple list, one per line."""
            
            response = openai_service.chat([
                {"role": "system", "content": "You generate practical classroom activity examples."},
                {"role": "user", "content": prompt}
            ], max_tokens=300, temperature=0.7)
            
            examples = [line.strip() for line in response.strip().split("\n") if line.strip()][:count]
            self._examples_cache[cache_key] = examples
        
        if replace:
            activity_customizer.add_examples_replace(act, examples)
            status = f"✅ Added {len(examples)} examples to {act.title}"
        else:
            activity_customizer.add_examples_merge(act, examples)
            status = f"✅ Added {len(examples)} examples to {act.title}"

        return {
            "type": "content",
            "subtype": "activity_update",
            "status": status,
            "section": "examples",
            "activity_markdown": self._clean_md(act.to_markdown()),
            "version": act.version,
        }

    def _handle_discussion_questions_request(self, act: Activity, user_text: str) -> Dict[str, Any]:
        """Handle discussion questions enhancement request using LLM generator"""
        replace = re.search(r"(?:replace|new|different|instead)", user_text.lower()) is not None
        
        # Parse dynamic count
        count = self._parse_dynamic_count(user_text, 5)
        
        # Get context for better tailoring
        grade = self.context.get("grade_level")
        subject = self.context.get("subject_area")
        
        # Check cache first
        cache_key = (act.title, grade or "", subject or "", count)
        if cache_key in self._questions_cache:
            questions = self._questions_cache[cache_key]
        else:
            # Generate questions using LLM
            prompt = f"""Generate {count} thoughtful discussion questions for this classroom activity.

Activity: {act.title}
Grade Level: {grade or "not specified"}
Subject: {subject or "not specified"}

Existing questions: {act.sections.get("discussion_questions", [])}

Return {count} discussion questions that promote reflection and deeper thinking. Format as a simple list, one per line."""
            
            response = openai_service.chat([
                {"role": "system", "content": "You generate thoughtful discussion questions for classroom activities."},
                {"role": "user", "content": prompt}
            ], max_tokens=300, temperature=0.7)
            
            questions = [line.strip() for line in response.strip().split("\n") if line.strip()][:count]
            self._questions_cache[cache_key] = questions
        
        if replace:
            activity_customizer.add_discussion_questions(act, questions, replace=True)
            status = f"✅ Added {len(questions)} discussion questions to {act.title}"
        else:
            activity_customizer.add_discussion_questions(act, questions, replace=False)
            status = f"✅ Added {len(questions)} discussion questions to {act.title}"

        return {
            "type": "content",
            "subtype": "activity_update",
            "status": status,
            "section": "discussion_questions",
            "activity_markdown": self._clean_md(act.to_markdown()),
            "version": act.version,
        }

    def _handle_materials_request(self, act: Activity, user_text: str) -> Dict[str, Any]:
        """Handle materials enhancement request using LLM generator"""
        replace = re.search(r"(?:replace|new|different|instead)", user_text.lower()) is not None
        
        # Parse dynamic count
        count = self._parse_dynamic_count(user_text, 5)
        
        # Get context for better tailoring
        grade = self.context.get("grade_level")
        subject = self.context.get("subject_area")
        
        # Check cache first
        cache_key = (act.title, grade or "", subject or "", count)
        if cache_key in self._materials_cache:
            materials = self._materials_cache[cache_key]
        else:
            # Generate materials using LLM
            prompt = f"""Generate {count} specific materials needed for this classroom activity.

Activity: {act.title}
Grade Level: {grade or "not specified"}
Subject: {subject or "not specified"}

Existing materials: {act.sections.get("materials", [])}

Return {count} concrete materials that teachers need. Format as a simple list, one per line."""
            
            response = openai_service.chat([
                {"role": "system", "content": "You generate lists of materials needed for classroom activities."},
                {"role": "user", "content": prompt}
            ], max_tokens=300, temperature=0.7)
            
            materials = [line.strip() for line in response.strip().split("\n") if line.strip()][:count]
            self._materials_cache[cache_key] = materials
        
        if replace:
            activity_customizer.add_materials(act, materials, replace=True)
            status = f"✅ Added {len(materials)} materials to {act.title}"
        else:
            activity_customizer.add_materials(act, materials, replace=False)
            status = f"✅ Added {len(materials)} materials to {act.title}"

        return {
            "type": "content",
            "subtype": "activity_update",
            "status": status,
            "section": "materials",
            "activity_markdown": self._clean_md(act.to_markdown()),
            "version": act.version,
        }

    def _handle_directions_request(self, act: Activity, user_text: str) -> Dict[str, Any]:
        """Handle directions enhancement request using LLM generator"""
        # Generate directions using the existing AI service
        prompt = f"""
        Generate clear, step-by-step directions for the activity: {act.title}
        
        Current activity context:
        - Overview: {act.sections.get('intro', '')}
        - Current directions: {act.sections.get('directions', [])}
        
        User request: {user_text}
        
        Provide numbered steps that are clear and actionable. Return as a JSON list of strings.
        """
        
        try:
            response = openai_service.generate_response(prompt)
            import json
            directions = json.loads(response)
            steps = directions if isinstance(directions, list) else [str(directions)]
        except:
            steps = [
                "Step 1: Prepare the materials",
                "Step 2: Explain the activity to participants",
                "Step 3: Facilitate the activity",
                "Step 4: Debrief and discuss"
            ]
        
        activity_customizer.set_directions(act, steps)
        
        return {
            "type": "content",
            "subtype": "activity_update",
            "status": f"✅ Updated directions for {act.title}",
            "section": "directions",
            "activity_markdown": self._clean_md(act.to_markdown()),
            "version": act.version,
        }

    def _handle_objectives_request(self, act: Activity, user_text: str) -> Dict[str, Any]:
        """Handle objectives enhancement request using LLM generator"""
        replace = re.search(r"(?:replace|new|different|instead)", user_text.lower()) is not None
        
        # Generate objectives using the existing AI service
        prompt = f"""
        Generate learning objectives for the activity: {act.title}
        
        Current activity context:
        - Overview: {act.sections.get('intro', '')}
        - Current objectives: {act.sections.get('objectives', [])}
        
        User request: {user_text}
        
        Provide clear, measurable learning objectives. Return as a JSON list of strings.
        """
        
        try:
            response = openai_service.generate_response(prompt)
            import json
            objectives = json.loads(response)
            objectives = objectives if isinstance(objectives, list) else [str(objectives)]
        except:
            objectives = [
                "Students will engage with the activity",
                "Students will learn from the experience",
                "Students will reflect on the outcomes"
            ]
        
        if replace:
            activity_customizer.add_objectives(act, objectives, replace=True)
            status = f"✅ Updated {len(objectives)} objectives for {act.title}"
        else:
            activity_customizer.add_objectives(act, objectives, replace=False)
            status = f"✅ Added {len(objectives)} objectives to {act.title}"

        return {
            "type": "content",
            "subtype": "activity_update",
            "status": status,
            "section": "objectives",
            "activity_markdown": self._clean_md(act.to_markdown()),
            "version": act.version,
        }

    def _handle_assessment_request(self, act: Activity, user_text: str) -> Dict[str, Any]:
        """Handle assessment enhancement request using LLM generator"""
        replace = re.search(r"(?:replace|new|different|instead)", user_text.lower()) is not None
        
        # Generate assessment using the existing AI service
        prompt = f"""
        Generate assessment methods for the activity: {act.title}
        
        Current activity context:
        - Overview: {act.sections.get('intro', '')}
        - Current assessment: {act.sections.get('assessment', [])}
        
        User request: {user_text}
        
        Provide assessment methods that align with the activity goals. Return as a JSON list of strings.
        """
        
        try:
            response = openai_service.generate_response(prompt)
            import json
            assessment = json.loads(response)
            assessment = assessment if isinstance(assessment, list) else [str(assessment)]
        except:
            assessment = [
                "Observe student participation",
                "Collect student feedback",
                "Review completed work"
            ]
        
        if replace:
            activity_customizer.add_assessment(act, assessment, replace=True)
            status = f"✅ Updated {len(assessment)} assessment methods for {act.title}"
        else:
            activity_customizer.add_assessment(act, assessment, replace=False)
            status = f"✅ Added {len(assessment)} assessment methods to {act.title}"

        return {
            "type": "content",
            "subtype": "activity_update",
            "status": status,
            "section": "assessment",
            "activity_markdown": self._clean_md(act.to_markdown()),
            "version": act.version,
        }


    # ---- Thin helpers expected by app.py ----
    def analyze_user_intent(self, user_text: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple intent detector aligned with app.py expectations.
        Returns one of: {"intent": "search" | "modify" | "help"} plus an echo.
        """
        text = (user_text or "").strip().lower()
        wants_modify = bool(re.search(
            r"\b(modify|change|update|edit|adapt|tweak|customize|shorten|condense|simplify|make\s+.*shorter|reduce|longer|increase|decrease|adjust|tailor)\b",
            text,
        )) or bool(
            re.search(
                r"(add|give|include|create).*(example|examples|discussion|question|questions|material|materials|direction|directions|step|steps|objective|objectives|assessment)",
                text,
            )
        )

        wants_help = bool(re.search(r"\b(help|how\s+do\s+i|what\s+can\s+you\s+do|capabilit(?:y|ies))\b", text))

        if wants_help:
            return {"intent": "help", "echo": user_text}
        if wants_modify and self.context.get("current_activity"):
            return {"intent": "modify", "echo": user_text}
        return {"intent": "search", "echo": user_text}

    def generate_follow_up_questions(self, search_results: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[str]:
        """Provide shortlist follow-ups the UI expects."""
        ideas = [
            "Open 1",
            "Open 2",
            "Show details for 1",
            "Add discussion questions to 1",
            "Write a short teacher script",
            "Build a downloadable lesson plan",
        ]
        return ideas[:3] if not search_results else ideas

    def get_contextual_help(self, conversation_stage: str = "") -> str:
        return (
            "Here’s how I can help:\n"
            "• Find activities by time, grade, or topic\n"
            "• Build ready-to-teach lesson plans\n"
            "• Customize activities (scripts, questions, handouts)\n"
            "• Answer teaching strategy questions\n\n"
            "Try: “show 15 minute check-in activities for grade 6”, "
            "or “add 5 discussion questions to the activity we opened.”"
        )

    @property
    def conversation_context(self) -> Dict[str, Any]:
        """app.py references conversation_manager.conversation_context."""
        return self.context

    # ---- Quick follow-up handlers (deterministic) ----
    def handle_related_request(self) -> Dict[str, Any]:
        act = self.context.get("current_activity")
        if not act:
            return {"type": "status", "message": "Open an activity first."}
        from data.processor import data_processor
        q = f"Similar to: {act.title}"
        try:
            dists, idxs = data_processor.search(q, top_k=8)
        except Exception:
            idxs = [[]]
        lines = [f"**Similar activities to _{act.title}_:**"]
        for i in (idxs[0][:8] if idxs and idxs[0] else []):
            row = data_processor.get_row(i)
            t = data_processor.safe_get(row, "Strategic Action") or "Untitled"
            tm = data_processor.safe_get(row, "Time") or ""
            lines.append(f"- {t}{f' ({tm})' if tm else ''}")
        return {
            "type": "content",
            "subtype": "related",
            "status": "✅ Here are similar activities.",
            "activity_markdown": "\n".join(lines),
        }

    def handle_shorten_request(self, target_minutes: int = 10) -> Dict[str, Any]:
        act = self.context.get("current_activity")
        if not act:
            return {"type": "status", "message": "Open an activity first."}
        steps = act.sections.get("directions", [])
        if not isinstance(steps, list):
            steps = [str(steps)]
        trimmed = [s for s in steps if str(s).strip()][:5]
        if not trimmed:
            trimmed = [
                "1) Briefly introduce the task (≤1 min).",
                "2) Model one quick example (≤1 min).",
                "3) Students do the core task (6–7 min).",
                "4) 1–2 volunteers share (1 min).",
            ]
        from services.activity_customizer import activity_customizer
        act = activity_customizer.set_directions(act, trimmed)
        act.upsert_text_section("time", f"{max(8, min(target_minutes, 12))} min")
        return {
            "type": "content",
            "subtype": "activity_update",
            "status": f"✅ Shortened to ~{target_minutes} minutes.",
            "activity_markdown": self._clean_md(act.to_markdown()),
            "version": act.version,
        }

    def handle_expand_request(self, target_minutes: int = 20) -> Dict[str, Any]:
        act = self.context.get("current_activity")
        if not act:
            return {"type": "status", "message": "Open an activity first."}
        steps = act.sections.get("directions", [])
        if not isinstance(steps, list):
            steps = [str(steps)]
        extension = [
            "• Warm-up hook (2–3 min).",
            "• Extended practice or second round (5–7 min).",
            "• Share-out + mini-reflection (3–5 min).",
            "• Exit ticket (1–2 min).",
        ]
        steps = steps + extension
        from services.activity_customizer import activity_customizer
        act = activity_customizer.set_directions(act, steps)
        act.upsert_text_section("time", f"{target_minutes}–{max(target_minutes+5, target_minutes)} min")
        return {
            "type": "content",
            "subtype": "activity_update",
            "status": f"✅ Expanded to ~{target_minutes} minutes with extensions.",
            "activity_markdown": self._clean_md(act.to_markdown()),
            "version": act.version,
        }

# Global instance
conversation_manager = ConversationManager()
