import sys, os
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import base64
import json
import uuid

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Load .env from project root (two levels up from src/api/main.py)
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        # Try loading from current directory
        load_dotenv()
        print("ℹ️  No .env file found, using system environment variables")
except ImportError:
    print("ℹ️  python-dotenv not installed, using system environment variables only")

# Silence the HF tokenizer fork warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src to path when running from project root (ensure package imports work)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.ai_services import openai_service  # ensure available before others

from config import config
from data.processor import data_processor
from services.db_only_conversation_engine import db_only_conversation_engine
from processing.document_processor import document_processor
print("🔎 main.py is using document_processor from:", document_processor.__class__.__module__)
print("🔎 build_docx defined in:", document_processor.build_docx.__code__.co_filename)
from services.content_retrieval import content_retrieval
from utils.serialization import strip_binary_fields

app = FastAPI(title="ClassroomGPT API")

# Enable CORS for local React dev servers
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    from services.ai_services import openai_service
    return {
        "ok": True,
        "port": 8787,
        "engine": "db_only",
        "openai_client": bool(openai_service.client)
    }


class ChatRequest(BaseModel):
    session: Dict[str, Any]
    message: str


class SessionRequest(BaseModel):
    session: Dict[str, Any] = {}


class ModifyRequest(BaseModel):
    session: Dict[str, Any]
    target: str
    changes: Dict[str, Any]


@app.on_event("startup")
def startup() -> None:
    data_processor.initialize(config.data_path)


def clean_text(text: str) -> str:
    """Clean text to ensure valid UTF-8 encoding"""
    if not text:
        return ""
    try:
        # Try to encode/decode to catch encoding issues
        return text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        return str(text).encode('ascii', errors='ignore').decode('ascii')

def _build_capabilities_payload(session: Dict[str, Any]) -> Dict[str, Any]:
    """Return a typed onboarding/capabilities payload for the UI."""
    preferred_time = session.get("preferred_time")
    qa = [
        {"label": "Find 10-min activities", "payload": "Find 10 minute activities"},
        {"label": "Build a lesson plan", "payload": "Create a 30-min lesson on empathy for Grade 6"},
        {"label": "Customize an activity", "payload": "Shorten Classmate Bingo to 10 minutes"},
        {"label": "Ask a teaching question", "payload": "How do I run quick check-ins?"}
    ]
    if isinstance(preferred_time, int):
        qa.insert(0, {"label": f"Find {preferred_time}-min activities", "payload": f"Find {preferred_time} minute activities"})

    return {
        "type": "capabilities",
        "content": (
            "Hi there! 👋 I can help you:\n\n"
            "• **Find Activities** by time, topic, or grade\n"
            "• **Create Lesson Plans** with materials & directions\n"
            "• **Customize** activities (shorten, change grade, add questions)\n"
            "• **Answer** teaching and classroom questions"
        ),
        "quick_actions": qa,
        "show_shortlist": False,
        "show_download": False,
        "debug": {"pipeline": ["intent_router"]}
    }

@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    print(f"🔍 [DEBUG] api: Chat request received: {req.message[:50]}...")
    session = req.session or {}

    # Increment per-turn counter for shortlist freshness
    session["turn_counter"] = session.get("turn_counter", -1) + 1

    # Ensure message history exists and append user message for context
    session.setdefault("messages", [])
    session["messages"].append({"role": "user", "content": req.message})

    # --- Unified flow: All requests go through process_message (handles everything)
    try:
        # Process through conversation engine (handles routing, modifications, searches, etc.)
        resp = db_only_conversation_engine.process_message(
            req.message,
            session
        )
        
        # Debug: Check last_row_index after processing (especially after followup_customize)
        if resp.get("type") == "lesson_plan":
            print(f"🔍 [DEBUG] After followup_customize, session last_row_index = {session.get('last_row_index')}")
            print(f"🔍 [DEBUG] Session last_opened_index = {session.get('last_opened_index')}")

    except Exception as e:
        import traceback
        print("💥 [ERROR] api: chat processing failed\n" + traceback.format_exc())
        resp = {
            "type": "response",
            "content": f"Sorry — I hit an internal error: {e}. Try rephrasing or start with a new search.",
            "activities": [],
            "show_shortlist": False,
            "error": str(e),
        }

    # Enhanced safety wrapper for non-dict responses
    if not isinstance(resp, dict):
        print("⚠️ [WARN] api: Engine returned non-dict/None; coercing to safe response")
        resp = {
            "ok": False,
            "message": "I had trouble interpreting that, but I kept your current activity. Try rephrasing or say 'open 1' to select an activity to modify.",
            "intent": None,
            "type": "response",
            "content": "I had trouble interpreting that, but I kept your current activity. Try rephrasing or say 'open 1' to select an activity to modify.",
            "activities": [],
            "show_shortlist": False
        }

    print(f"🔍 [DEBUG] api: Engine response type: {resp.get('type', 'unknown')}")

    # Clean the response content to ensure valid UTF-8
    if "content" in resp:
        resp["content"] = clean_text(resp["content"])
    
    # Clean activities data
    if "activities" in resp:
        for activity in resp["activities"]:
            if "title" in activity:
                activity["title"] = clean_text(activity["title"])
            if "block" in activity:
                activity["block"] = clean_text(activity["block"])

    # Update session with shortlist indices/results if present
    if resp.get("show_shortlist") and resp.get("activities"):
        # Ensure we have valid activities with row_index
        valid_activities = [a for a in resp["activities"] if a.get("row_index") is not None]
        
        if valid_activities:
            indices = [int(a.get("row_index")) for a in valid_activities]
            titles = [clean_text(str(a.get("title", "")).strip()) for a in valid_activities]
            blocks = [clean_text(str(a.get("block", "")).strip()) for a in valid_activities]
            
            # Store all three consistently
            session["last_indices"] = indices
            session["last_titles"] = titles  # Store titles separately
            session["last_results"] = valid_activities  # Store full activity dicts for LLM router
            session["last_blocks"] = blocks  # Store rich descriptions for React
            
            print(f"🔍 [DEBUG] api: Updated session with {len(indices)} activities")
            print(f"🔍 [DEBUG] api: last_indices = {indices}")
            print(f"🔍 [DEBUG] api: last_titles = {titles}")
            print(f"🔍 [DEBUG] api: last_results = {len(valid_activities)} activity dicts")

    # Always append a chat message (assistant role) using chat_markdown/chat_message when present
    assistant_text = resp.get("chat_markdown") or resp.get("chat_message") or resp.get("message")
    if assistant_text:
        session["messages"].append({"role": "assistant", "content": assistant_text})
    elif resp.get("content"):
        # Fallback to content if no chat_markdown/chat_message
        session["messages"].append({"role": "assistant", "content": str(resp.get("content", ""))})
    
    # Remove doc_bytes from lesson plan responses - use download endpoint instead
    if resp.get("type") == "lesson_plan" and resp.get("lesson_plan", {}).get("doc_bytes"):
        lp = resp["lesson_plan"].copy()
        # Remove doc_bytes from response, client should use download endpoint
        if "doc_bytes" in lp:
            del lp["doc_bytes"]
        resp["lesson_plan"] = lp
        print("🔍 [DEBUG] api: Removed doc_bytes from lesson plan response")
    
    # Sanitize the response to remove any binary data
    print("🔍 [DEBUG] api: Sanitizing response")
    sanitized_resp = strip_binary_fields(resp)
    
    # Also sanitize the session data to remove any pandas objects
    print("🔍 [DEBUG] api: Sanitizing session data")
    from helpers.session_sanitize import sanitize_session_data
    sanitized_session = sanitize_session_data(session)
    
    # Add session_id to response for frontend
    sanitized_resp["session"] = sanitized_session
    
    # Safety guard: If handler forgot to set type=lesson_plan but provided content_md, fix it
    if sanitized_resp.get("view") == "lesson_plan" and sanitized_resp.get("content_md") and sanitized_resp.get("type") != "lesson_plan":
        print("🔍 [DEBUG] api: Converting response to lesson_plan format")
        sanitized_resp["type"] = "lesson_plan"
        # Minimal lesson_plan payload so UI buttons work
        sanitized_resp.setdefault("lesson_plan", {
            "title": sanitized_resp.get("title") or "Lesson Plan",
            "filename": f"{(sanitized_resp.get('title') or 'Lesson Plan').replace('/', '-')}.docx",
            "activity": {},
            "row_index": (sanitized_session.get("active_focus") or {}).get("activity_id"),
            "doc_bytes": None
        })
    
    # Minimal heuristic: show download only if content has any classroom sections
    if "content" in sanitized_resp:
        c = (sanitized_resp["content"] or "").lower()
        sanitized_resp["show_download"] = any(k in c for k in ["directions", "materials", "objective", "examples"])
    
    # Add debug info to help identify which server responded
    engine_from_resp = None
    if isinstance(sanitized_resp, dict):
        engine_from_resp = (
            (sanitized_resp.get("debug") or {}).get("engine")
            or sanitized_resp.get("_engine")
            or sanitized_resp.get("engine")
        )

    debug_info = {
        "engine": engine_from_resp or "db_only",
        "pipeline": (sanitized_resp.get("debug") or {}).get("pipeline"),
        "port": 8787,
        "count": len(sanitized_resp.get("activities", [])),
        "has_display_ids": any("display_id" in a for a in sanitized_resp.get("activities", []))
    }
    
    return {"response": sanitized_resp, "session": sanitized_session, "debug": debug_info}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat endpoint for word-by-word display"""
    
    async def generate_stream():
        session = req.session or {}
        session_id = session.get("session_id") or str(uuid.uuid4())
        session["session_id"] = session_id
        
        # Increment per-turn counter
        session["turn_counter"] = session.get("turn_counter", -1) + 1
        
        # Ensure message history exists and append user message
        session.setdefault("messages", [])
        session["messages"].append({"role": "user", "content": req.message})
        
        try:
            # Process message
            result = db_only_conversation_engine.process_message(
                req.message,
                session
            )
            
            # Get content to stream
            content = result.get("content", "")
            
            # Stream the response word by word
            words = content.split()
            
            for i, word in enumerate(words):
                chunk = {
                    "word": word + " ",
                    "is_last": i == len(words) - 1,
                    "session_id": session_id
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send final metadata
            from helpers.session_sanitize import sanitize_session_data, strip_binary_fields
            
            # Clean activities data
            if "activities" in result:
                for activity in result["activities"]:
                    if "title" in activity:
                        activity["title"] = clean_text(activity["title"])
            
            # Sanitize response
            sanitized_result = strip_binary_fields(result)
            sanitized_session = sanitize_session_data(session)
            
            final_chunk = {
                "done": True,
                "metadata": {
                    "type": sanitized_result.get("type"),
                    "activities": sanitized_result.get("activities", []),
                    "lesson_plan": sanitized_result.get("lesson_plan"),
                    "show_shortlist": sanitized_result.get("show_shortlist", False)
                },
                "session": sanitized_session,
                "session_id": session_id
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/download/{row_index}")
def download_activity(row_index: int):
    """Download activity as Word document"""
    print(f"🔍 [DEBUG] api: Download request for row {row_index}")
    try:
        row = data_processor.get_row(row_index)
        title = data_processor.safe_get(row, "Strategic Action") or "Activity"
        text, _ = content_retrieval.text_from_supabase_or_fallback(row, allow_web=True)
        directions = document_processor.make_directions(text, title) if text.strip() else ""
        _, filename, doc_bytes, _ = document_processor.format_activity_block(row, directions)
        
        print(f"🔍 [DEBUG] api: Generated document {filename} ({len(doc_bytes)} bytes)")
        return Response(
            content=doc_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        print(f"🔍 [DEBUG] api: Download failed for row {row_index}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/activities/{row_index}")
def get_activity(row_index: int, req: Optional[SessionRequest] = None) -> Dict[str, Any]:
    """Get activity metadata for editing (no binary data)"""
    print(f"🔍 [DEBUG] api: Activity metadata request for row {row_index}")
    
    # Extract session from request body
    session = req.session if req else {}
    
    # ✅ FIX: Check if we have a modified version in session first
    if session and session.get('last_modified_activity') and session.get('last_row_index') == row_index:
        print(f"✅ [SESSION] Returning modified activity from session for row {row_index}")
        modified_activity = session['last_modified_activity']
        
        # Merge Source Link into Additional Resources
        source_link = (modified_activity.get('Source Link') or modified_activity.get('Source') or '').strip()
        additional_resources = (modified_activity.get('Additional Resources') or '').strip()
        
        if source_link:
            if additional_resources:
                modified_activity['Additional Resources'] = f"{additional_resources}\n\nSource: {source_link}"
            else:
                modified_activity['Additional Resources'] = f"Source: {source_link}"
        
        title = modified_activity.get('Activity Name') or modified_activity.get('Title') or 'Activity'
        filename = f"{title.replace('/', '-')}.docx"
        block = document_processor.format_activity_template(modified_activity)
        
        result = {
            "title": title,
            "content": block,
            "filename": filename,
            "activity": modified_activity,
        }
        
        sanitized_result = strip_binary_fields(result)
        print(f"🔍 [DEBUG] api: Returning modified activity with {len(modified_activity.get('Directions', ''))} char directions")
        return sanitized_result
    
    try:
        from data.content_cache import content_cache
        
        # Get pre-generated lesson plan from cache
        lesson_plan_md, summary = content_cache.get_generated(row_index)
        
        if lesson_plan_md:
            # Parse markdown to dict - this will have the correct field names
            activity = document_processor.markdown_to_dict(lesson_plan_md)
            print(f"✅ [DEBUG] api: Loaded activity from cache with keys: {list(activity.keys())}")
        else:
            # Fallback: generate on the fly if not cached
            print(f"⚠️ [WARN] api: No cached lesson plan for row {row_index}, generating...")
            row = data_processor.get_row(row_index)
            title = data_processor.safe_get(row, "Strategic Action")
            text, _ = content_retrieval.text_from_supabase_or_fallback(row, allow_web=True)
            directions = document_processor.make_directions(text, title) if text.strip() else ""
            block, filename, _, activity = document_processor.format_activity_block(row, directions)
        
        # ✅ FIX: Merge Source Link into Additional Resources for the edit form
        source_link = (activity.get('Source Link') or activity.get('Source') or '').strip()
        additional_resources = (activity.get('Additional Resources') or '').strip()
        
        if source_link:
            if additional_resources:
                # Both exist - combine them
                activity['Additional Resources'] = f"{additional_resources}\n\nSource: {source_link}"
            else:
                # Only source exists - use it
                activity['Additional Resources'] = f"Source: {source_link}"
        
        print(f"🔍 [DEBUG] api: Additional Resources = {activity.get('Additional Resources', '')[:100]}...")
        
        # Get the title and filename
        title = activity.get('Activity Name') or activity.get('Title') or 'Activity'
        filename = f"{title.replace('/', '-')}.docx"
        
        # Create block representation for display
        block = document_processor.format_activity_template(activity)
        
        result = {
            "title": title,
            "content": block,
            "filename": filename,
            "activity": activity,  # This now has Source Link in Additional Resources!
        }
        
        # Sanitize the response
        sanitized_result = strip_binary_fields(result)
        print(f"🔍 [DEBUG] api: Returning activity with fields: {list(activity.keys())}")
        return sanitized_result
        
    except Exception as e:
        import traceback
        print(f"💥 [ERROR] api: Activity metadata request failed: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/activities/modify")
def modify_activity(req: ModifyRequest) -> Dict[str, Any]:
    print(f"🔍 [DEBUG] api: Modify activity request for {req.target}")
    # Simulated modify; echo changes and return updated session
    session = req.session or {}
    message = f"Modify activity {req.target} with {req.changes}"
    resp = db_only_conversation_engine.process_message(message, session)
    
    # Sanitize the response
    sanitized_resp = strip_binary_fields(resp)
    print("🔍 [DEBUG] api: Returning sanitized modify response")
    return {"response": sanitized_resp, "session": session}


class BuildDocRequest(BaseModel):
    activity: Dict[str, Any]


@app.post("/documents/build")
def build_document(req: BuildDocRequest) -> Dict[str, Any]:
    print("🔍 [DEBUG] api: Build document request")
    try:
        activity = req.activity or {}
        # Build document from provided activity fields
        blob = document_processor.build_docx_strategic_action(activity)
        title = activity.get("Activity Name", "Lesson Plan")
        safe = (title or "Lesson Plan").strip()[:80]
        filename = f"{safe}.docx"
        
        result = {
            "filename": filename,
            "doc_bytes": base64.b64encode(blob).decode('utf-8'),
        }
        
        # Note: This endpoint still returns base64 encoded bytes for backward compatibility
        # In a full implementation, you might want to use a separate download endpoint here too
        print(f"🔍 [DEBUG] api: Built document {filename} ({len(blob)} bytes)")
        return result
    except Exception as e:
        print(f"🔍 [DEBUG] api: Build document failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class CondenseRequest(BaseModel):
    activity: Dict[str, Any]


@app.post("/documents/condense")
def condense_document(req: CondenseRequest) -> Dict[str, Any]:
    print("🔍 [DEBUG] api: Condense document request")
    try:
        activity = req.activity or {}
        compact = document_processor.condense_activity(activity)
        
        result = {"activity": compact}
        # Sanitize the response
        sanitized_result = strip_binary_fields(result)
        print("🔍 [DEBUG] api: Returning sanitized condense response")
        return sanitized_result
    except Exception as e:
        print(f"🔍 [DEBUG] api: Condense document failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class ModifyFreeformRequest(BaseModel):
    activity: Dict[str, Any]
    instruction: str


@app.post("/documents/modify")
def modify_document(req: ModifyFreeformRequest) -> Dict[str, Any]:
    print(f"🔍 [DEBUG] api: Modify document request with instruction: {req.instruction[:50]}...")
    try:
        activity = req.activity or {}
        instruction = req.instruction or ""
        updated = document_processor.modify_activity_with_instruction(activity, instruction)
        
        # Also return a full markdown render of the updated plan
        content_md = document_processor.legacy_to_markdown(updated)
        result = {"activity": updated, "content_md": content_md}
        # Sanitize the response
        sanitized_result = strip_binary_fields(result)
        print("🔍 [DEBUG] api: Returning sanitized modify response")
        return sanitized_result
    except Exception as e:
        print(f"🔍 [DEBUG] api: Modify document failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

