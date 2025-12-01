"""
Activity customization and enhancement
"""
from typing import Dict, List, Any
from processing.document_processor import Activity, _norm_text

class ActivityCustomizer:
    """Handles structured activity enhancements"""

    def add_examples_replace(self, act: Activity, examples: List[str]) -> Activity:
        """Replace the examples section with new examples (idempotent)."""
        act.replace_list_section("examples", examples)
        return act

    def add_examples_merge(self, act: Activity, examples: List[str]) -> Activity:
        """Merge examples into the examples section (deduped)."""
        act.upsert_list_section("examples", examples)
        return act

    def set_directions(self, act: Activity, steps: List[str]) -> Activity:
        """Replace the directions section (idempotent)."""
        act.replace_list_section("directions", steps)
        return act

    def add_discussion_questions(self, act: Activity, qs: List[str], replace: bool = False) -> Activity:
        """Add or replace discussion questions."""
        if replace:
            act.replace_list_section("discussion_questions", qs)
        else:
            act.upsert_list_section("discussion_questions", qs)
        return act

    def add_materials(self, act: Activity, materials: List[str], replace: bool = False) -> Activity:
        """Add or replace materials."""
        if replace:
            act.replace_list_section("materials", materials)
        else:
            act.upsert_list_section("materials", materials)
        return act

    def set_intro(self, act: Activity, intro: str) -> Activity:
        """Replace the intro/overview section."""
        act.upsert_text_section("intro", intro)
        return act

    def add_objectives(self, act: Activity, objectives: List[str], replace: bool = False) -> Activity:
        """Add or replace objectives."""
        if replace:
            act.replace_list_section("objectives", objectives)
        else:
            act.upsert_list_section("objectives", objectives)
        return act

    def add_assessment(self, act: Activity, assessment: List[str], replace: bool = False) -> Activity:
        """Add or replace assessment items."""
        if replace:
            act.replace_list_section("assessment", assessment)
        else:
            act.upsert_list_section("assessment", assessment)
        return act

    def set_notes(self, act: Activity, notes: str) -> Activity:
        """Replace the notes section."""
        act.upsert_text_section("notes", notes)
        return act

    # --- UI adapter methods expected by app.py ---
    def generate_customization_options(self, flat_activity: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Return UI-ready suggestion buckets used by app.py action buttons.
        """
        name = (flat_activity.get("Activity Name") or "This activity").strip()
        return [
            {
                "title": "Discussion Questions",
                "description": f"Add questions to deepen reflection in {name}.",
                "suggestions": ["Add 5 discussion questions", "Replace discussion questions"],
            },
            {
                "title": "Materials",
                "description": f"List concrete materials needed for {name}.",
                "suggestions": ["Add 5 materials", "Replace materials"],
            },
            {
                "title": "Directions",
                "description": f"Clarify step-by-step directions for {name}.",
                "suggestions": ["Rewrite directions with numbered steps"],
            },
            {
                "title": "Objectives",
                "description": f"Add measurable goals for {name}.",
                "suggestions": ["Add 3 learning objectives"],
            },
            {
                "title": "Assessment",
                "description": f"Add quick formative checks for {name}.",
                "suggestions": ["Add 3 assessment ideas"],
            },
        ]

    def suggest_related_activities(self, flat_activity: Dict[str, str], ctx: Dict[str, Any]) -> List[str]:
        """Lightweight related ideas for the UI list."""
        base = (flat_activity.get("Activity Name") or "Activity").split(":")[0].strip()
        return [
            f"{base} — quick pair-share version",
            f"{base} — small group rotation",
            f"{base} — extended 30-minute deep-dive",
        ]

    def customize_activity(self, flat_activity: Dict[str, str], suggestion: str, ctx: Dict[str, Any]) -> Dict[str, str]:
        """
        Adapter used by app.py buttons. Operates on the legacy flat dict shape.
        """
        out = dict(flat_activity or {})
        text = (suggestion or "").lower()

        if "discussion" in text or "question" in text:
            import re as _re
            m = _re.search(r"\b(\d{1,2})\b", text)
            count = int(m.group(1)) if m else 5
            qs = [f"Discussion question {i+1}." for i in range(count)]
            existing = [s.strip() for s in (out.get("Directions") or "").split("\n") if s.strip()]
            block = "\n".join(existing + ["", "**Discussion Questions:**"] + [f"- {q}" for q in qs]).strip()
            out["Directions"] = block

        elif "material" in text:
            import re as _re
            m = _re.search(r"\b(\d{1,2})\b", text)
            count = int(m.group(1)) if m else 5
            mats = [f"{out.get('Activity Name','Activity')} material {i+1}" for i in range(count)]
            current = out.get("Materials") or ""
            prefix = (current + "\n") if current.strip() else ""
            out["Materials"] = prefix + "\n".join(f"- {m}" for m in mats)

        elif "objective" in text:
            import re as _re
            m = _re.search(r"\b(\d{1,2})\b", text)
            count = int(m.group(1)) if m else 3
            objs = [f"Students will be able to (SWBAT) do outcome {i+1}." for i in range(count)]
            current = out.get("Objective") or ""
            prefix = (current + "\n") if current.strip() else ""
            out["Objective"] = prefix + "\n".join(f"- {o}" for o in objs)

        elif any(k in text for k in ("direction", "step", "rewrite")):
            steps = [
                "1) Prepare materials and display the prompt.",
                "2) Introduce the task and model one example.",
                "3) Students work (pair/small group) as you circulate.",
                "4) Share-out and brief debrief.",
                "5) Optional extension or exit ticket.",
            ]
            out["Directions"] = "\n".join(steps)

        elif any(k in text for k in ("assessment", "check")):
            checks = [
                "- Thumbs-up/middle/down pulse check",
                "- Exit ticket: one insight + one question",
                "- Quick rubric: participation & accuracy",
            ]
            current = out.get("Directions") or ""
            out["Directions"] = (current + "\n\n**Assessment ideas:**\n" + "\n".join(checks)).strip()

        elif __import__('re').search(r"\b(shorten|condense|make .*shorter|10\s*min)\b", text):
            out["Time"] = "8–10 min"
            core = [
                "1) 60-sec intro & model.",
                "2) Students do core task (6–7 min).",
                "3) Quick share-out (1–2 min).",
            ]
            out["Directions"] = "\n".join(core)

        elif __import__('re').search(r"\b(expand|extend|make .*longer|20\s*min|30\s*min)\b", text):
            base = [line for line in (out.get("Directions") or "").splitlines() if line.strip()]
            base += [
                "• Warm-up hook (2–3 min).",
                "• Extra practice/round (5–7 min).",
                "• Exit ticket (1–2 min).",
            ]
            out["Directions"] = "\n".join(base)
            out["Time"] = "20–25 min"

        else:
            out["Activity Name"] = f"{out.get('Activity Name','Activity')} — {suggestion}"

        return out

# Global instance
activity_customizer = ActivityCustomizer()
