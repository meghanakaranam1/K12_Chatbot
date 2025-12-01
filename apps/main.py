"""
Main ClassroomGPT application - Simple, fast interface
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
# ✅ NEW: use DB-only conversation engine
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
print("🔍 [DEBUG] apps/main.py: Setting up page configuration")
st.set_page_config(page_title="🎓 ClassroomGPT", layout="wide")
st.title("🎓 ClassroomGPT")
st.caption("I'll suggest activities first — reply with 'open 2', a title fragment, or ask for something new.")
print("🔍 [DEBUG] apps/main.py: Page configuration complete")

# ---------- Sidebar Configuration ----------
with st.sidebar:
    st.header("Data & Options")
    config.data_path = st.text_input("CSV path", config.data_path)
    ALLOW_WEB_FALLBACK = st.checkbox("Allow web fallback (Source Link/HTML)", True)
    STRICT_TIME_FILTER = st.checkbox("Respect time if user mentions duration", True)
    ALLOW_FUZZY_FALLBACK = st.checkbox("If too strict, allow fuzzy (+50%)", True)
    st.caption(f"Supabase: {'✅' if config.is_supabase_configured else '❌'} | Bucket: {config.supabase_bucket}")

# ---------- Session State Management ----------
print("🔍 [DEBUG] apps/main.py: Initializing session state")
if "sessions" not in st.session_state:
    print("🔍 [DEBUG] apps/main.py: Loading sessions from file")
    st.session_state.sessions = json.loads(config.sessions_file.read_text("utf-8")) if config.sessions_file.exists() else {}

if "current_session" not in st.session_state or st.session_state.current_session not in st.session_state.sessions:
    print("🔍 [DEBUG] apps/main.py: Creating new session")
    sid = str(uuid.uuid4())[:8]
    st.session_state.sessions[sid] = {
        "title": "New chat",
        "messages": [{"role":"assistant","content":"Hi! Tell me a topic, time, and goals — I'll show a shortlist."}],
        "last_results": [],
        "last_indices": [],
        "open_indices": [],
        # ✅ NEW: fields the DB-only engine expects
        "active_focus": {},
        "overrides": {},
    }
    st.session_state.current_session = sid
    print(f"🔍 [DEBUG] apps/main.py: Created session {sid}")

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
    print("🔍 [DEBUG] apps/main.py: Saving sessions to disk")
    try:
        data = st.session_state.sessions
        config.sessions_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"🔍 [DEBUG] apps/main.py: Saved {len(data)} sessions to {config.sessions_file}")
    except Exception as e:
        print(f"🔍 [DEBUG] apps/main.py: Failed to save sessions: {e}")
        pass

def stable_activity_id(row_index: int, title: str) -> str:
    """Generate stable activity ID for feedback tracking"""
    key = f"{row_index}|{title}".encode("utf-8", "ignore")
    return "a:" + hashlib.sha1(key).hexdigest()[:12]

# ---------- Initialize Data Processor ----------
@st.cache_resource(show_spinner=False)
def initialize_data():
    """Initialize data processor"""
    print("🔍 [DEBUG] apps/main.py: Initializing data processor")
    data_processor.initialize(config.data_path)
    print("🔍 [DEBUG] apps/main.py: Data processor initialized")

print("🔍 [DEBUG] apps/main.py: Calling initialize_data()")
initialize_data()
print("🔍 [DEBUG] apps/main.py: Data initialization complete")

# ---------- Core Query Processing (legacy; no longer used directly) ----------
def query_bot(user_input: str, top_k: int = None, max_results: int = None) -> List[Dict[str, Any]]:
    """
    Legacy search function kept for backward compatibility.
    DB-only engine below now handles all user messages.
    """
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
            
            # Format activity block
            block, filename, doc_bytes, activity = document_processor.format_activity_block(row, directions)
            
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
    """Generate simple preview label with improved formatting"""
    row = data_processor.get_row(dataset_index)
    
    # Activity name (first line only)
    raw_title = data_processor.safe_get(row, "Strategic Action") or "Untitled Activity"
    title = raw_title.splitlines()[0].strip()
    
    # Time field
    time_str = data_processor.safe_get(row, "Time") or data_processor.safe_get(row, "Time to implement")
    
    # Description - try multiple fields
    desc = (data_processor.safe_get(row, "Short Description") or 
            data_processor.safe_get(row, "Notes") or 
            data_processor.safe_get(row, "Objective") or "").strip()
    desc = desc.replace("\n", " ").strip()
    
    # Benefit - try multiple fields
    benefit = (data_processor.safe_get(row, "Benefit") or
               data_processor.safe_get(row, "Why it works") or
               data_processor.safe_get(row, "Rationale") or "").strip()
    benefit = benefit.replace("\n", " ").strip()
    
    # Fallback benefit extraction
    if not benefit and desc:
        match = re.search(r"(helps|support|improve|build|develop|encourage|engage|promote|aid|foster)\b.*", desc, flags=re.I)
        if match:
            benefit = match.group(0).rstrip(".")
        else:
            benefit = " ".join(desc.split()[:10])
    
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
    print(f"🔍 [DEBUG] apps/main.py: Adding {role} message: {content[:50]}...")
    msg_id = str(uuid.uuid4())[:8]
    cur()["messages"].append({"role": role, "content": content, "msg_id": msg_id, "attachments": attachments or []})
    print(f"🔍 [DEBUG] apps/main.py: Message added with ID: {msg_id}")
    save_sessions()
    return msg_id

# ---------- Shortlist Display ----------
def show_shortlist(indices: List[int]):
    """Show shortlist of activities - original simple interface"""
    print(f"🔍 [DEBUG] apps/main.py: Showing shortlist with {len(indices)} activities")
    labels = [preview_label(i) for i in indices]
    cur()["last_indices"] = indices
    cur()["last_results"] = labels
    print(f"🔍 [DEBUG] apps/main.py: Generated {len(labels)} preview labels")
    
    lines = ["**Here are some activities I found:**\n"]
    for pos, (i, label) in enumerate(zip(indices, labels), start=1):
        title = label.splitlines()[0]
        rest = "\n".join(label.splitlines()[1:])
        lines.append(f"**{pos}. {title}**")
        if rest:
            lines.append(rest)
        print(f"🔍 [DEBUG] apps/main.py: Added activity {pos}: {title}")
    
    lines.append("\n_Reply with **open 1** or a title fragment (e.g., 'goal setting')._")
    md = "\n\n".join(lines)
    
    print("🔍 [DEBUG] apps/main.py: Displaying shortlist")
    with st.chat_message("assistant"):
        st.markdown(md)
    add_msg("assistant", md)

# ---------- Activity Expansion ----------
def expand_indices(dataset_indices: List[int], user_query: str):
    """Expand and display selected activities"""
    print(f"🔍 [DEBUG] apps/main.py: Expanding {len(dataset_indices)} activities for query: {user_query[:50]}...")
    if not dataset_indices:
        print("🔍 [DEBUG] apps/main.py: No dataset indices provided")
        add_msg("assistant", "I didn't catch which option you wanted. Try a number or the activity name.")
        return
    
    rendered_blocks = []
    atts_meta = []
    atts_blobs = []
    chosen = []
    
    with st.chat_message("assistant"):
        with st.spinner(f"Building {len(dataset_indices)} plan(s)…"):
            for di in dataset_indices[:3]:
                print(f"🔍 [DEBUG] apps/main.py: Processing activity {di}")
                row = data_processor.get_row(di)
                title = data_processor.safe_get(row, "Strategic Action") or "Classroom Activity"
                print(f"🔍 [DEBUG] apps/main.py: Activity title: {title}")
                
                # Get text content
                print("🔍 [DEBUG] apps/main.py: Retrieving text content")
                text, _ = content_retrieval.text_from_supabase_or_fallback(row, allow_web=ALLOW_WEB_FALLBACK)
                
                # Generate directions
                print("🔍 [DEBUG] apps/main.py: Generating directions")
                directions = document_processor.make_directions(text, title) if text.strip() else ""
                
                # Format activity block
                print("🔍 [DEBUG] apps/main.py: Formatting activity block")
                block, filename, blob, activity = document_processor.format_activity_block(row, directions)
                
                # Feedback tracking
                activity_name = filename.replace('.docx', '')
                aid = stable_activity_id(di, activity_name)
                print(f"🔍 [DEBUG] apps/main.py: Tracking activity rendered: {activity_name}")
                track_activity_rendered(aid, activity_name)
                
                st.success(activity_name)
                st.markdown(block)
                
                # Download button
                print(f"🔍 [DEBUG] apps/main.py: Adding download button for {filename}")
                clicked = st.download_button(
                    label=f"📥 Download {filename}",
                    data=blob,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
                
                # Feedback tracking for download
                if clicked:
                    print(f"🔍 [DEBUG] apps/main.py: Download clicked for {activity_name}")
                    record_download_click(aid, activity_name, filename)
                
                # Inline edit form - original interface
                print("🔍 [DEBUG] apps/main.py: Setting up edit form")
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
                print(f"🔍 [DEBUG] apps/main.py: Added activity {di} to results")
    
    if rendered_blocks:
        print(f"🔍 [DEBUG] apps/main.py: Persisting {len(rendered_blocks)} activity blocks")
        msg_id = add_msg("assistant", "\n\n".join(rendered_blocks), attachments=atts_meta)
        vol().setdefault("attachments", {})[msg_id] = atts_blobs
    cur()["open_indices"] = chosen
    print(f"🔍 [DEBUG] apps/main.py: Expansion complete, {len(chosen)} activities processed")

# ---------- Command Handling ----------
def try_selection_or_command(text: str) -> bool:
    """Handle simple commands like 'open 2'"""
    print(f"🔍 [DEBUG] apps/main.py: Checking for selection/command in: {text}")
    # open N
    match = re.match(r"^(open|show)\s+(\d+)\b", text.strip().lower())
    if match and cur()["last_indices"]:
        n = int(match.group(2)) - 1
        print(f"🔍 [DEBUG] apps/main.py: Command match found, index: {n}")
        if 0 <= n < len(cur()["last_indices"]):
            print(f"🔍 [DEBUG] apps/main.py: Valid index, expanding activity {cur()['last_indices'][n]}")
            expand_indices([cur()["last_indices"][n]], text)
            return True
        else:
            print(f"🔍 [DEBUG] apps/main.py: Index {n} out of range (0-{len(cur()['last_indices'])-1})")
    else:
        print("🔍 [DEBUG] apps/main.py: No command match or no last_indices")
    return False

# ---------- Target Resolution ----------
def resolve_targets(target_strings: List[str], labels: List[str]) -> List[int]:
    """Resolve target activities from user input"""
    print(f"🔍 [DEBUG] apps/main.py: Resolving targets from {len(target_strings)} strings against {len(labels)} labels")
    if not labels or not target_strings:
        print("🔍 [DEBUG] apps/main.py: No labels or target strings provided")
        return []
    
    titles = [(l.splitlines()[0]).strip() for l in labels]
    title_lc = [t.lower() for t in titles]
    print(f"🔍 [DEBUG] apps/main.py: Extracted {len(titles)} titles")
    
    # Simple keyword matching
    best = {}
    for raw in target_strings:
        q = raw.strip().lower()
        print(f"🔍 [DEBUG] apps/main.py: Processing target string: {q}")
        
        # Handle numeric references
        nums = [int(n)-1 for n in re.findall(r"(?<![a-z0-9])#?\s*(\d{1,2})(?:st|nd|rd|th)?(?![a-z0-9])", q)]
        print(f"🔍 [DEBUG] apps/main.py: Found numeric references: {nums}")
        for n in nums:
            if 0 <= n < len(titles):
                best[n] = max(best.get(n, 0), 1.0)
                print(f"🔍 [DEBUG] apps/main.py: Added numeric match for index {n}")
        
        # Simple keyword matching
        for i, title in enumerate(title_lc):
            if any(word in title for word in q.split() if len(word) > 2):
                best[i] = max(best.get(i, 0), 0.8)
                print(f"🔍 [DEBUG] apps/main.py: Added keyword match for index {i}: {title}")
    
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    picks = [i for i, sc in ranked if sc >= 0.5]
    print(f"🔍 [DEBUG] apps/main.py: Ranked matches: {ranked}, picks: {picks}")
    
    seen, out = set(), []
    for i in picks:
        if 0 <= i < len(labels) and i not in seen:
            seen.add(i)
            out.append(i)
    print(f"🔍 [DEBUG] apps/main.py: Final resolved targets: {out[:3]}")
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
print("🔍 [DEBUG] apps/main.py: Setting up chat input")
user_text = st.chat_input("Type your message…")
if user_text:
    print(f"🔍 [DEBUG] apps/main.py: User input received: {user_text[:50]}...")
    with st.chat_message("user"):
        st.markdown(user_text)
    add_msg("user", user_text)
    print("🔍 [DEBUG] apps/main.py: User message added to session")
    
    # Quick command handling
    print("🔍 [DEBUG] apps/main.py: Checking for quick commands")
    if try_selection_or_command(user_text):
        print("🔍 [DEBUG] apps/main.py: Quick command handled, stopping")
        st.stop()
    
    # If we already showed a shortlist, see if the message refers to one of them
    print("🔍 [DEBUG] apps/main.py: Checking for target resolution")
    if cur()["last_indices"] and cur()["last_results"]:
        picks = resolve_targets([user_text], cur()["last_results"]) or []
        if picks:
            print(f"🔍 [DEBUG] apps/main.py: Resolved targets: {picks}")
            ds = [cur()["last_indices"][p] for p in picks if 0 <= p < len(cur()["last_indices"])]
            if ds:
                print(f"🔍 [DEBUG] apps/main.py: Expanding indices: {ds}")
                expand_indices(ds, user_text)
                st.stop()
    
    # ✅ NEW: Route ALL messages through the DB-only engine
    print("🔍 [DEBUG] apps/main.py: Processing through conversation engine")
    with st.chat_message("assistant"):
        with st.spinner("Finding good options…"):
            response = db_only_conversation_engine.process_message(user_text, cur())
    print(f"🔍 [DEBUG] apps/main.py: Engine response type: {response.get('type', 'unknown')}")
    
    # Render the engine response text
    content = response.get("content", "")
    if content:
        print(f"🔍 [DEBUG] apps/main.py: Rendering response content: {content[:50]}...")
        with st.chat_message("assistant"):
            st.markdown(content)
        add_msg("assistant", content)
    
    # Show shortlist if present
    if response.get("show_shortlist") and response.get("activities"):
        print(f"🔍 [DEBUG] apps/main.py: Showing shortlist with {len(response['activities'])} activities")
        indices = [r["row_index"] for r in response["activities"]]
        cur()["last_indices"] = indices
        show_shortlist(indices)
    else:
        # keep coherence even if no shortlist
        cur()["last_indices"] = response.get("activities", []) and [r["row_index"] for r in response["activities"]] or []
        print("🔍 [DEBUG] apps/main.py: No shortlist to show")

# ---------- Sidebar: Chat Management ----------
with st.sidebar:
    st.header("💬 Chats")
    if st.button("➕ New Chat", use_container_width=True):
        sid = str(uuid.uuid4())[:8]
        st.session_state.sessions[sid] = {
            "title": "New chat",
            "messages": [{"role":"assistant","content":"Hi! Tell me a topic, time, and goals — I'll show a shortlist."}],
            "last_results": [], "last_indices": [], "open_indices": [],
            # ✅ NEW: keep these here too
            "active_focus": {},
            "overrides": {},
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
