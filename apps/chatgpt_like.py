"""
"""
import os, io, re, json, uuid, pathlib, difflib, hashlib
from typing import List, Dict, Any, Tuple

# Silence the HF tokenizer fork warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
print("🔍 [DEBUG] apps/chatgpt_like.py: Setting up page configuration")
st.set_page_config(page_title="🎓 ClassroomGPT - ChatGPT-like", layout="wide")
st.title("🎓 ClassroomGPT")
st.caption("Ask me anything! I can help with activities, answer questions, create content, and manage your data.")
print("🔍 [DEBUG] apps/chatgpt_like.py: Page configuration complete")

# ---------- Sidebar Configuration ----------
with st.sidebar:
    st.header("⚙️ Settings")
    config.data_path = st.text_input("CSV path", config.data_path)
    ALLOW_WEB_FALLBACK = st.checkbox("Allow web fallback", True)
    STRICT_TIME_FILTER = st.checkbox("Respect time constraints", True)
    ALLOW_FUZZY_FALLBACK = st.checkbox("Allow fuzzy time matching", True)
    
    st.write("---")
    st.header("🔧 Capabilities")
    st.markdown("""
    **I can help you with:**
    - 🔍 **Find activities**: "Find team building activities"
    - ➕ **Add activities**: "Add a new math activity"
    - 🗑️ **Delete activities**: "Delete activity #3"
    - ✏️ **Modify activities**: "Change the time for activity #2"
    - 📊 **Show data**: "Show me all activities"
    - 📝 **Create content**: "Write a lesson plan about fractions"
    - ❓ **Answer questions**: "What is photosynthesis?"
    - 💬 **General chat**: "Help me with my homework"
    """)

# ---------- Session State Management ----------
print("🔍 [DEBUG] apps/chatgpt_like.py: Initializing session state")
if "sessions" not in st.session_state:
    print("🔍 [DEBUG] apps/chatgpt_like.py: Loading sessions from file")
    st.session_state.sessions = json.loads(config.sessions_file.read_text("utf-8")) if config.sessions_file.exists() else {}

if "current_session" not in st.session_state or st.session_state.current_session not in st.session_state.sessions:
    print("🔍 [DEBUG] apps/chatgpt_like.py: Creating new session")
    sid = str(uuid.uuid4())[:8]
    st.session_state.sessions[sid] = {
        "title": "New chat",
        "messages": [{"role":"assistant","content":"Hi! I'm your AI teaching assistant. I can help you with classroom activities, answer questions, create content, and manage your data. What would you like to do today?"}],
        "last_results": [],
        "last_indices": [],
        "open_indices": [],
        "overrides": {},
    }
    st.session_state.current_session = sid
    print(f"🔍 [DEBUG] apps/chatgpt_like.py: Created session {sid}")

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
    print("🔍 [DEBUG] apps/chatgpt_like.py: Initializing data processor")
    data_processor.initialize(config.data_path)
    print("🔍 [DEBUG] apps/chatgpt_like.py: Data processor initialized")

print("🔍 [DEBUG] apps/chatgpt_like.py: Calling initialize_data()")
initialize_data()
print("🔍 [DEBUG] apps/chatgpt_like.py: Data initialization complete")

# ---------- Core Query Processing (for activity search) ----------
def query_bot(user_input: str, top_k: int = None, max_results: int = None) -> List[Dict[str, Any]]:
    """Main query processing for activity search"""
    if top_k is None:
        top_k = config.top_k
    if max_results is None:
        max_results = config.max_results
    
    # Get time constraint from query
    asked_minutes = time_processor.desired_time_from_query(user_input)
    strict_set = time_processor.indices_with_time_constraint(asked_minutes, fuzzy=False) if STRICT_TIME_FILTER else None
    
    # Search for similar content
    distances, indices = data_processor.search(user_input, top_k)
    picked = indices[0]
    
    def collect_results(allowed_indices: set = None) -> List[Dict[str, Any]]:
        results = []
        for i in picked:
            if len(results) >= max_results:
                break
            if allowed_indices is not None and i not in allowed_indices:
                continue
            
            row = data_processor.get_row(i)
            title = data_processor.safe_get(row, "Strategic Action")
            
            # Get text content
            text, _ = content_retrieval.text_from_supabase_or_fallback(row, allow_web=ALLOW_WEB_FALLBACK)
            
            # Generate directions
            directions = document_processor.make_directions(text, title) if text.strip() else ""
            
            # Apply in-session overrides (e.g., modified time)
            ov = cur().get("overrides", {}).get(int(i), {})
            if ov.get("Time"):
                # Inject override before formatting
                row = row.copy()
                row["Time"] = ov["Time"]
                row["Time to implement"] = ov["Time"]
            
            # Format activity block
            block, filename, doc_bytes, activity = document_processor.format_activity_block(row, directions)
            if ov.get("Time"):
                activity["Time"] = ov["Time"]
            
            results.append({
                "row_index": int(i),
                "title": title,
                "block": block,
                "filename": filename,
                "docbytes": doc_bytes,
                "activity": activity
            })
        
        return results
    
    # Try strict filtering first
    results = collect_results(strict_set)
    
    # Try fuzzy filtering if no results
    if (not results) and STRICT_TIME_FILTER and ALLOW_FUZZY_FALLBACK and asked_minutes is not None:
        fuzzy_set = time_processor.indices_with_time_constraint(asked_minutes, fuzzy=True)
        results = collect_results(fuzzy_set)
    
    # Fallback to no time filtering
    if not results:
        results = collect_results(None)
    
    return results

# ---------- Preview Label Generation ----------
def preview_label(dataset_index: int) -> str:
    """Generate preview strictly from DB fields (no LLM) with improved formatting."""
    row = data_processor.get_row(dataset_index)

    # Title
    raw_title = data_processor.safe_get(row, "Strategic Action") or "Untitled Activity"
    title = raw_title.splitlines()[0].strip()

    # Time
    time_str = (data_processor.safe_get(row, "Time") or
                data_processor.safe_get(row, "Time to implement") or "").strip()

    # Description - try multiple fields
    desc = (data_processor.safe_get(row, "Short Description") or
            data_processor.safe_get(row, "Notes") or
            data_processor.safe_get(row, "Objective") or "").replace("\n", " ").strip()
    if not desc:
        directions = (data_processor.safe_get(row, "Directions") or "").replace("\n", " ").strip()
        words = directions.split()
        desc = " ".join(words[:20]) + ("…" if len(words) > 20 else "")

    # Benefit - try multiple fields
    benefit = (data_processor.safe_get(row, "Benefit") or
               data_processor.safe_get(row, "Why it works") or
               data_processor.safe_get(row, "Rationale") or "").strip()
    benefit = benefit.replace("\n", " ").strip()

    # Build formatted output
    lines = [f"### {title}"]
    if time_str:
        lines.append(f"⏱️ {time_str}")
    if desc:
        lines.append(desc)
    if benefit:
        lines.append(f"**Benefit:** {benefit}")
    lines.append("---")

    return "\n".join(lines)

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
    labels = [preview_label(i) for i in indices]
    cur()["last_indices"] = indices
    cur()["last_results"] = labels
    
    # Build markdown once so we can both display and persist it
    lines = []
    if conversational_response:
        lines.append(conversational_response)
    lines.append("**Here are the activities I found:**")
    for pos, (i, label) in enumerate(zip(indices, labels), start=1):
        title = label.splitlines()[0]
        rest = "\n".join(label.splitlines()[1:])
        lines.append(f"**{pos}. {title}**")
        if rest:
            lines.append(rest)
    lines.append("\n_You can say 'open 1' to see details, or ask me to modify, delete, or add activities._")
    shortlist_md = "\n\n".join(lines)

    # Display now
    with st.chat_message("assistant"):
        st.markdown(shortlist_md)
    
    # Persist to chat history so it remains visible on next turns
    add_msg("assistant", shortlist_md)

# ---------- Activity Expansion ----------
def expand_indices(dataset_indices: List[int], user_query: str):
    """Expand and display selected activities"""
    if not dataset_indices:
        add_msg("assistant", "I didn't catch which option you wanted. Try a number or the activity name.")
        return
    
    rendered_blocks = []
    atts_meta = []
    atts_blobs = []
    chosen = []
    
    with st.chat_message("assistant"):
        with st.spinner(f"Building {len(dataset_indices)} plan(s)…"):
            for di in dataset_indices[:3]:
                row = data_processor.get_row(di)
                title = data_processor.safe_get(row, "Strategic Action") or "Classroom Activity"
                
                # Get text content
                text, _ = content_retrieval.text_from_supabase_or_fallback(row, allow_web=ALLOW_WEB_FALLBACK)
                
                # Generate directions
                directions = document_processor.make_directions(text, title) if text.strip() else ""
                
                # Format activity block
                block, filename, blob, activity = document_processor.format_activity_block(row, directions)
                
                # Feedback tracking
                activity_name = filename.replace('.docx', '')
                aid = stable_activity_id(di, activity_name)
                track_activity_rendered(aid, activity_name)
                
                st.success(activity_name)
                st.markdown(block)
                
                # Download button
                clicked = st.download_button(
                    label=f"📥 Download {filename}",
                    data=blob,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
                
                # Feedback tracking for download
                if clicked:
                    record_download_click(aid, activity_name, filename)
                
                # Inline edit form
                with st.expander("Edit before download"):
                    with st.form(key=f"edit_{di}"):
                        a = activity.copy()
                        a["Activity Name"] = st.text_input("Activity Name", a.get("Activity Name", ""))
                        a["Objective"] = st.text_area("Objective", a.get("Objective", ""), height=80)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            a["Advance preparation"] = st.text_area("Advance preparation", a.get("Advance preparation", ""), height=80)
                            a["Materials"] = st.text_area("Materials", a.get("Materials", ""), height=80)
                            a["Student Materials"] = st.text_area("Student Materials", a.get("Student Materials", ""), height=80)
                        with col2:
                            a["Time"] = st.text_input("Time", a.get("Time", ""))
                            a["Introduction"] = st.text_area("Introduction", a.get("Introduction", ""), height=80)
                            a["Additional Resources"] = st.text_area("Additional Resources", a.get("Additional Resources", ""), height=80)
                        
                        a["Source Link"] = st.text_input("Source Link (URL)", a.get("Source Link", ""))
                        a["Directions"] = st.text_area("Directions (1. with a., b. sub-steps)", a.get("Directions", ""), height=160)
                        
                        if st.form_submit_button("Build & Download"):
                            fn = re.sub(r"[^\w\s-]", "", a.get("Activity Name") or "Activity").strip() or "Activity"
                            fn = re.sub(r"\s+", " ", fn)[:80] + ".docx"
                            ok2 = st.download_button(
                                "📄 Download",
                                document_processor.build_docx(a),
                                fn,
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                            if ok2:
                                record_download_click(aid, activity_name, fn)
                
                rendered_blocks.append(f"### {activity_name}\n\n{block}")
                atts_meta.append({"file_name": filename, "mime":"application/vnd.openxmlformats-officedocument.wordprocessingml.document"})
                atts_blobs.append({"data": blob})
                chosen.append(di)
    
    if rendered_blocks:
        msg_id = add_msg("assistant", "\n\n".join(rendered_blocks), attachments=atts_meta)
        vol().setdefault("attachments", {})[msg_id] = atts_blobs
    cur()["open_indices"] = chosen

# ---------- Command Handling ----------
def try_selection_or_command(text: str) -> bool:
    """Handle simple commands like 'open 2'"""
    # open N
    match = re.match(r"^(open|show)\s+(\d+)\b", text.strip().lower())
    if match and cur()["last_indices"]:
        n = int(match.group(2)) - 1
        if 0 <= n < len(cur()["last_indices"]):
            expand_indices([cur()["last_indices"][n]], text)
            return True
    return False

# ---------- Target Resolution ----------
def resolve_targets(target_strings: List[str], labels: List[str]) -> List[int]:
    """Resolve target activities from user input"""
    if not labels or not target_strings:
        return []
    
    titles = [(l.splitlines()[0]).strip() for l in labels]
    title_lc = [t.lower() for t in titles]
    
    # Simple keyword matching
    best = {}
    for raw in target_strings:
        q = raw.strip().lower()
        
        # Handle numeric references
        nums = [int(n)-1 for n in re.findall(r"(?<![a-z0-9])#?\s*(\d{1,2})(?:st|nd|rd|th)?(?![a-z0-9])", q)]
        for n in nums:
            if 0 <= n < len(titles):
                best[n] = max(best.get(n, 0), 1.0)
        
        # Simple keyword matching
        for i, title in enumerate(title_lc):
            if any(word in title for word in q.split() if len(word) > 2):
                best[i] = max(best.get(i, 0), 0.8)
    
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    picks = [i for i, sc in ranked if sc >= 0.5]
    
    seen, out = set(), []
    for i in picks:
        if 0 <= i < len(labels) and i not in seen:
            seen.add(i)
            out.append(i)
    return out[:3]

# ---------- Chat History Rendering ----------
for m in cur()["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        atts = m.get("attachments") or []
        if atts:
            blobs = vol().get("attachments", {}).get(m.get("msg_id"), [])
            for i, meta in enumerate(atts):
                data = blobs[i]["data"] if (i < len(blobs) and isinstance(blobs[i], dict)) else b""
                st.download_button(
                    label=f"📥 Download {meta.get('file_name', 'download')}",
                    data=data,
                    file_name=meta.get("file_name", "download.bin"),
                    mime=meta.get("mime", "application/octet-stream"),
                    key=f"dl_hist_{m['msg_id']}_{i}",
                    use_container_width=True,
                )

# Feedback prompt trigger
maybe_prompt_for_feedback()

# ---------- Main Chat Loop ----------
print("🔍 [DEBUG] apps/chatgpt_like.py: Setting up chat input")
user_text = st.chat_input("Ask me anything! I can help with activities, answer questions, create content...")
if user_text:
    print(f"🔍 [DEBUG] apps/chatgpt_like.py: User input received: {user_text[:50]}...")
    with st.chat_message("user"):
        st.markdown(user_text)
    add_msg("user", user_text)
    print("🔍 [DEBUG] apps/chatgpt_like.py: User message added to session")
    
    # Quick command handling (for backward compatibility)
    print("🔍 [DEBUG] apps/chatgpt_like.py: Checking for quick commands")
    if try_selection_or_command(user_text):
        print("🔍 [DEBUG] apps/chatgpt_like.py: Quick command handled, stopping")
        st.stop()
    
    # Use the conversation engine for all messages (including activity details)
    # This handles both new searches and requests for specific activities
    print("🔍 [DEBUG] apps/chatgpt_like.py: Processing through conversation engine")
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = db_only_conversation_engine.process_message(user_text, cur())
    print(f"🔍 [DEBUG] apps/chatgpt_like.py: Engine response type: {response.get('type', 'unknown')}")
    
    # Handle the response
    if response["type"] == "response":
        print(f"🔍 [DEBUG] apps/chatgpt_like.py: Rendering response content: {response['content'][:50]}...")
        st.markdown(response["content"])
        add_msg("assistant", response["content"])
        
        # Show shortlist if activities were found
        if response.get("show_shortlist") and response.get("activities"):
            print(f"🔍 [DEBUG] apps/chatgpt_like.py: Showing shortlist with {len(response['activities'])} activities")
            show_shortlist([r["row_index"] for r in response["activities"]], "")
            
        # Handle special actions
        if "new_activity" in response:
            print("🔍 [DEBUG] apps/chatgpt_like.py: Handling new activity creation")
            # Show the new activity for confirmation
            activity = response["new_activity"]
            st.markdown("**New Activity Created:**")
            st.json(activity)
            
            if st.button("Save to Database"):
                st.success("Activity saved! (Note: In a full implementation, this would save to your database)")
        
        if "delete_activity" in response:
            print("🔍 [DEBUG] apps/chatgpt_like.py: Handling activity deletion")
            # Show delete confirmation
            activity = response["delete_activity"]
            if st.button("Yes, Delete This Activity"):
                st.success(f"Activity '{activity['title']}' deleted! (Note: In a full implementation, this would remove from database)")

    elif response["type"] == "lesson_plan":
        print("🔍 [DEBUG] apps/chatgpt_like.py: Handling lesson plan response")
        # Handle detailed lesson plan display
        lesson_plan = response["lesson_plan"]
        # Build a single markdown block so we can persist it
        block_md = f"{response['content']}\n\n{lesson_plan['content']}"
        
        # Display now
        st.markdown(block_md)
        
        # Download + Edit (always show), and persist attachments to history
        atts_meta = []
        atts_blobs = []
        if lesson_plan.get("doc_bytes"):
            print("🔍 [DEBUG] apps/chatgpt_like.py: Adding download button for lesson plan")
            clicked = st.download_button(
                label="📄 Download Lesson Plan",
                data=lesson_plan["doc_bytes"],
                file_name=lesson_plan["filename"],
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
            atts_meta.append({
                "file_name": lesson_plan["filename"],
                "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            })
            atts_blobs.append({"data": lesson_plan["doc_bytes"]})

        # Edit before download (rebuild a structured activity for editing)
        print("🔍 [DEBUG] apps/chatgpt_like.py: Setting up edit form for lesson plan")
        try:
            ri = int(lesson_plan.get("row_index"))
            row = data_processor.get_row(ri)
            title = data_processor.safe_get(row, "Strategic Action") or lesson_plan.get("title") or "Classroom Activity"
            # Recompute directions and activity so the form is populated
            text, _ = content_retrieval.text_from_supabase_or_fallback(row, allow_web=ALLOW_WEB_FALLBACK)
            directions = document_processor.make_directions(text, title) if text.strip() else ""
            _, _, _, activity = document_processor.format_activity_block(row, directions)
        except Exception:
            activity = {"Activity Name": lesson_plan.get("title", "Classroom Activity"), "Objective": "", "Materials": "", "Time": "", "Introduction": "", "Additional Resources": "", "Source Link": "", "Directions": ""}

        with st.expander("Edit before download"):
            with st.form(key=f"edit_lp_{lesson_plan.get('row_index','x')}"):
                a = activity.copy()
                a["Activity Name"] = st.text_input("Activity Name", a.get("Activity Name", ""))
                a["Objective"] = st.text_area("Objective", a.get("Objective", ""), height=80)
                col1, col2 = st.columns(2)
                with col1:
                    a["Advance preparation"] = st.text_area("Advance preparation", a.get("Advance preparation", ""), height=80)
                    a["Materials"] = st.text_area("Materials", a.get("Materials", ""), height=80)
                    a["Student Materials"] = st.text_area("Student Materials", a.get("Student Materials", ""), height=80)
                with col2:
                    a["Time"] = st.text_input("Time", a.get("Time", ""))
                    a["Introduction"] = st.text_area("Introduction", a.get("Introduction", ""), height=80)
                    a["Additional Resources"] = st.text_area("Additional Resources", a.get("Additional Resources", ""), height=80)
                a["Source Link"] = st.text_input("Source Link (URL)", a.get("Source Link", ""))
                a["Directions"] = st.text_area("Directions (1. with a., b. sub-steps)", a.get("Directions", ""), height=160)
                if st.form_submit_button("Build & Download"):
                    fn = re.sub(r"[^\w\s-]", "", a.get("Activity Name") or "Activity").strip() or "Activity"
                    fn = re.sub(r"\s+", " ", fn)[:80] + ".docx"
                    blob = document_processor.build_docx(a)
                    ok2 = st.download_button("📄 Download", blob, fn, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                    if ok2:
                        record_download_click(stable_activity_id(lesson_plan.get("row_index", 0), a.get("Activity Name", "Activity")), a.get("Activity Name", "Activity"), fn)
        
        
        # Track activity rendered
        print("🔍 [DEBUG] apps/chatgpt_like.py: Tracking activity rendered")
        track_activity_rendered(lesson_plan["row_index"], lesson_plan["title"])
        
        # Persist the full lesson plan so it stays on screen in history (with attachment for download)
        if atts_meta:
            print("🔍 [DEBUG] apps/chatgpt_like.py: Persisting lesson plan with attachments")
            msg_id = add_msg("assistant", block_md, attachments=atts_meta)
            vol().setdefault("attachments", {})[msg_id] = atts_blobs
        else:
            print("🔍 [DEBUG] apps/chatgpt_like.py: Persisting lesson plan without attachments")
            add_msg("assistant", block_md)

# ---------- Sidebar: Chat Management ----------
with st.sidebar:
    st.header("💬 Chats")
    if st.button("➕ New Chat", use_container_width=True):
        sid = str(uuid.uuid4())[:8]
        st.session_state.sessions[sid] = {
            "title": "New chat",
            "messages": [{"role":"assistant","content":"Hi! I'm your AI teaching assistant. I can help you with classroom activities, answer questions, create content, and manage your data. What would you like to do today?"}],
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
