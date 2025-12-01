"""
Feedback tracking and user interaction
"""
import json
import streamlit as st
from pathlib import Path
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Feedback file path
FEEDBACK_FILE = Path("feedback_events.jsonl")

def track_activity_rendered(activity_id: str, activity_name: str):
    """Track when an activity is rendered"""
    print(f"🔍 [DEBUG] feedback: Tracking activity rendered - {activity_name} ({activity_id})")
    event = {
        "event_type": "activity_rendered",
        "activity_id": activity_id,
        "activity_name": activity_name,
        "timestamp": st.session_state.get("_timestamp", "unknown")
    }
    _write_feedback_event(event)

def record_download_click(activity_id: str, activity_name: str, filename: str):
    """Track when a download button is clicked"""
    print(f"🔍 [DEBUG] feedback: Recording download click - {activity_name} ({filename})")
    event = {
        "event_type": "download_clicked",
        "activity_id": activity_id,
        "activity_name": activity_name,
        "filename": filename,
        "timestamp": st.session_state.get("_timestamp", "unknown")
    }
    _write_feedback_event(event)

def maybe_prompt_for_feedback():
    """Show feedback prompt if conditions are met"""
    # Simple implementation - can be enhanced
    if st.session_state.get("_show_feedback_prompt", False):
        with st.sidebar:
            st.info("💬 How was your experience? We'd love your feedback!")

def _write_feedback_event(event: Dict[str, Any]):
    """Write feedback event to file"""
    print(f"🔍 [DEBUG] feedback: Writing event to file: {event['event_type']}")
    try:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"🔍 [DEBUG] feedback: Event written successfully to {FEEDBACK_FILE}")
    except Exception as e:
        print(f"🔍 [DEBUG] feedback: Failed to write event: {e}")
        pass  # Fail silently
