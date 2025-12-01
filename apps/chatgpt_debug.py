"""
Debug version of ChatGPT-like ClassroomGPT with better error handling
"""
import os, io, re, json, uuid, pathlib, difflib, hashlib
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our modular components
from config import config
from data.processor import data_processor
from data.time_processor import time_processor
from services.ai_services import openai_service, supabase_service
from services.content_retrieval import content_retrieval
from processing.document_processor import document_processor
from ui.feedback import track_activity_rendered, record_download_click, maybe_prompt_for_feedback
from services.db_only_conversation_engine import db_only_conversation_engine

def _rerun():
    """Handle Streamlit rerun compatibility"""
    try:
        st.rerun()  # modern Streamlit (stable)
    except AttributeError:
        try:
            st.experimental_rerun()  # older Streamlit
        except Exception:
            pass

# ---------- Page Configuration ----------
st.set_page_config(page_title="🎓 ClassroomGPT - Debug Mode", layout="wide")
st.title("🎓 ClassroomGPT - Debug Mode")
st.caption("Debug version with better error handling. Ask me anything!")

# ---------- Sidebar Configuration ----------
with st.sidebar:
    st.header("⚙️ Debug Info")
    st.write(f"**Data Path:** {config.data_path}")
    st.write(f"**OpenAI Configured:** {config.is_openai_configured}")
    st.write(f"**Supabase Configured:** {config.is_supabase_configured}")
    
    # Test data loading
    try:
        data_processor.initialize(config.data_path)
        st.success("✅ Data loaded successfully")
        st.write(f"**Activities Count:** {len(data_processor.df) if data_processor.df is not None else 'Unknown'}")
    except Exception as e:
        st.error(f"❌ Data loading error: {e}")
    
    st.write("---")
    st.header("🔧 Capabilities")
    st.markdown("""
    **I can help you with:**
    - 🔍 **Find activities**: "Find team building activities"
    - ❓ **Answer questions**: "What is photosynthesis?"
    - 📝 **Create content**: "Write a lesson plan about fractions"
    - 💬 **General chat**: "Help me with my homework"
    """)

# ---------- Session State Management ----------
if "sessions" not in st.session_state:
    st.session_state.sessions = json.loads(config.sessions_file.read_text("utf-8")) if config.sessions_file.exists() else {}

if "current_session" not in st.session_state or st.session_state.current_session not in st.session_state.sessions:
    sid = str(uuid.uuid4())[:8]
    st.session_state.sessions[sid] = {
        "title": "New chat",
        "messages": [{"role":"assistant","content":"Hi! I'm your AI teaching assistant in debug mode. I can help you with classroom activities, answer questions, create content, and manage your data. What would you like to do today?"}],
        "last_results": [],
        "last_indices": [],
        "open_indices": [],
    }
    st.session_state.current_session = sid

# Volatile large blobs per-session (not saved to disk)
if "_volatile" not in st.session_state:
    st.session_state._volatile = {}

def cur():
    """Get current session"""
    return st.session_state.sessions[st.session_state.current_session]

def vol():
    """Get volatile session data"""
    return st.session_state._volatile.setdefault(st.session_state.current_session, {"attachments":{}})

def save_sessions():
    """Save sessions to disk"""
    try:
        data = st.session_state.sessions
        config.sessions_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def stable_activity_id(row_index: int, title: str) -> str:
    """Generate stable activity ID for feedback tracking"""
    key = f"{row_index}|{title}".encode("utf-8", "ignore")
    return "a:" + hashlib.sha1(key).hexdigest()[:12]

# ---------- Initialize Data Processor ----------
@st.cache_resource(show_spinner=False)
def initialize_data():
    """Initialize data processor"""
    try:
        data_processor.initialize(config.data_path)
        return True
    except Exception as e:
        st.error(f"Data initialization error: {e}")
        return False

data_loaded = initialize_data()

# ---------- Simple Preview Label Generation ----------
def preview_label(dataset_index: int) -> str:
    """Generate simple preview label"""
    try:
        row = data_processor.get_row(dataset_index)
        
        # Activity name (first line only)
        raw_title = data_processor.safe_get(row, "Strategic Action") or "Untitled Activity"
        title = raw_title.splitlines()[0].strip()
        
        # Time field
        time_str = data_processor.safe_get(row, "Time") or data_processor.safe_get(row, "Time to implement")
        header = f"{title} ({time_str})" if time_str else title
        
        # Description
        desc = (data_processor.safe_get(row, "Short Description") or data_processor.safe_get(row, "Notes") or "").strip()
        desc = desc.replace("\n", " ").strip()
        
        return f"**{header}**\n\n{desc}"
    except Exception as e:
        return f"**Activity {dataset_index}**\n\nError loading details: {e}"

# ---------- Message Management ----------
def add_msg(role: str, content: str, attachments: list = None) -> str:
    """Add message to current session"""
    msg_id = str(uuid.uuid4())[:8]
    cur()["messages"].append({"role": role, "content": content, "msg_id": msg_id, "attachments": attachments or []})
    save_sessions()
    return msg_id

# ---------- Enhanced Shortlist Display ----------
def show_shortlist(indices: List[int], conversational_response: str = ""):
    """Show shortlist with conversational response"""
    try:
        labels = [preview_label(i) for i in indices]
        cur()["last_indices"] = indices
        cur()["last_results"] = labels
        
        with st.chat_message("assistant"):
            if conversational_response:
                st.markdown(conversational_response)
            
            st.markdown("**Here are the activities I found:**")
            
            for pos, (i, label) in enumerate(zip(indices, labels), start=1):
                title = label.splitlines()[0]
                rest = "\n".join(label.splitlines()[1:])
                st.markdown(f"**{pos}. {title}**")
                if rest:
                    st.markdown(rest)
            
            st.markdown("\n_You can say 'open 1' to see details, or ask me to modify, delete, or add activities._")
        
        # Add to message history
        full_response = conversational_response + "\n\n**Here are the activities I found:**\n\n"
        for pos, (i, label) in enumerate(zip(indices, labels), start=1):
            title = label.splitlines()[0]
            full_response += f"**{pos}. {title}**\n"
        
        add_msg("assistant", full_response)
    except Exception as e:
        st.error(f"Error showing shortlist: {e}")

def show_shortlist_from_results(results: List[Dict], conversational_response: str = ""):
    """Show shortlist from conversation engine results"""
    try:
        with st.chat_message("assistant"):
            if conversational_response:
                st.markdown(conversational_response)
            
            st.markdown("**Here are the activities I found:**")
            
            for pos, result in enumerate(results, start=1):
                title = result.get("title", "Untitled Activity")
                time_str = result.get("time", "")
                desc = result.get("description", "")
                
                # Display activity preview
                if time_str:
                    st.markdown(f"**{pos}. {title}** ({time_str})")
                else:
                    st.markdown(f"**{pos}. {title}**")
                
                if desc:
                    st.markdown(f"_{desc}_")
                
                st.markdown("")  # Add spacing
            
            st.markdown("_You can say 'open 1' to see details, or ask me to modify, delete, or add activities._")
        
        # Store results for later reference
        cur()["last_indices"] = [r["row_index"] for r in results]
        cur()["last_results"] = [r["title"] for r in results]
        
        # Add to message history
        full_response = conversational_response + "\n\n**Here are the activities I found:**\n\n"
        for pos, result in enumerate(results, start=1):
            title = result.get("title", "Untitled Activity")
            full_response += f"**{pos}. {title}**\n"
        
        add_msg("assistant", full_response)
    except Exception as e:
        st.error(f"Error showing shortlist from results: {e}")

# ---------- Command Handling ----------
def try_selection_or_command(text: str) -> bool:
    """Handle simple commands like 'open 2'"""
    # open N
    match = re.match(r"^(open|show)\s+(\d+)\b", text.strip().lower())
    if match and cur()["last_indices"]:
        n = int(match.group(2)) - 1
        if 0 <= n < len(cur()["last_indices"]):
            # Simple activity display
            try:
                row_index = cur()["last_indices"][n]
                row = data_processor.get_row(row_index)
                title = data_processor.safe_get(row, "Strategic Action")
                
                with st.chat_message("assistant"):
                    st.success(f"**{title}**")
                    st.markdown(f"**Time:** {data_processor.safe_get(row, 'Time', 'Not specified')}")
                    st.markdown(f"**Objective:** {data_processor.safe_get(row, 'Objective', 'Not specified')}")
                    st.markdown(f"**Materials:** {data_processor.safe_get(row, 'Materials', 'Not specified')}")
                
                add_msg("assistant", f"**{title}** - Details displayed above")
            except Exception as e:
                st.error(f"Error displaying activity: {e}")
            return True
    return False

# ---------- Chat History Rendering ----------
for m in cur()["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Feedback prompt trigger
maybe_prompt_for_feedback()

# ---------- Main Chat Loop ----------
user_text = st.chat_input("Ask me anything! I can help with activities, answer questions, create content...")
if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    add_msg("user", user_text)
    
    # Quick command handling (for backward compatibility)
    if try_selection_or_command(user_text):
        st.stop()
    
    # Use the simple conversation engine for all other messages
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = db_only_conversation_engine.process_message(user_text, cur())
            except Exception as e:
                response = {
                    "type": "response",
                    "content": f"I encountered an error: {e}\n\nPlease try rephrasing your request or ask me something else!",
                    "activities": [],
                    "show_shortlist": False
                }
    
    # Handle the response
    if response["type"] == "response":
        # Show shortlist if activities were found
        if response.get("show_shortlist") and response.get("activities"):
            show_shortlist_from_results(response["activities"], response["content"])
        else:
            # Regular response without shortlist
            st.markdown(response["content"])
            add_msg("assistant", response["content"])

# ---------- Sidebar: Chat Management ----------
with st.sidebar:
    st.header("💬 Chats")
    if st.button("➕ New Chat", use_container_width=True):
        sid = str(uuid.uuid4())[:8]
        st.session_state.sessions[sid] = {
            "title": "New chat",
            "messages": [{"role":"assistant","content":"Hi! I'm your AI teaching assistant in debug mode. I can help you with classroom activities, answer questions, create content, and manage your data. What would you like to do today?"}],
            "last_results": [], "last_indices": [], "open_indices": []
        }
        st.session_state.current_session = sid
        save_sessions()
        st.rerun()
    
    cur_title = cur().get("title", "New chat")
    new_title = st.text_input("Rename current chat", cur_title)
    if new_title.strip() and new_title != cur_title:
        cur()["title"] = new_title.strip()
        save_sessions()
    
    st.write("---")
    for sid, payload in list(st.session_state.sessions.items()):
        cols = st.columns([1, 0.22])
        if cols[0].button(payload.get("title") or f"Chat {sid}", key=f"switch_{sid}", use_container_width=True):
            st.session_state.current_session = sid
            save_sessions()
            st.rerun()
        if cols[1].button("🗑", key=f"del_{sid}"):
            if len(st.session_state.sessions) > 1:
                del st.session_state.sessions[sid]
                if st.session_state.current_session == sid:
                    st.session_state.current_session = next(iter(st.session_state.sessions))
                save_sessions()
                st.rerun()
