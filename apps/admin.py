"""
Admin dashboard for ClassroomGPT
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import config
from data.processor import data_processor

st.set_page_config(page_title="🎓 ClassroomGPT Admin", layout="wide")
st.title("🎓 ClassroomGPT Admin Dashboard")

# Initialize data
try:
    data_processor.initialize(config.data_path)
    st.success("✅ Data loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")

# Data Overview
st.header("📊 Data Overview")
if data_processor.df is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Activities", len(data_processor.df))
    with col2:
        st.metric("Columns", len(data_processor.df.columns))
    with col3:
        st.metric("Search Columns", len(config.search_columns))

# Configuration
st.header("⚙️ Configuration")
st.json({
    "OpenAI Model": config.openai_model,
    "Embedding Model": config.embed_model,
    "Data Path": config.data_path,
    "Top K": config.top_k,
    "Max Results": config.max_results,
    "Supabase Configured": config.is_supabase_configured,
    "OpenAI Configured": config.is_openai_configured
})

# Data Preview
st.header("📋 Data Preview")
if data_processor.df is not None:
    st.dataframe(data_processor.df.head(10))

# Session Management
st.header("💬 Sessions")
sessions_file = Path("sessions.json")
if sessions_file.exists():
    with open(sessions_file, 'r') as f:
        sessions = json.load(f)
    
    st.write(f"Total sessions: {len(sessions)}")
    
    for session_id, session_data in sessions.items():
        with st.expander(f"Session {session_id}: {session_data.get('title', 'Untitled')}"):
            st.write(f"Messages: {len(session_data.get('messages', []))}")
            st.write(f"Last results: {len(session_data.get('last_results', []))}")
else:
    st.write("No sessions found")

# Feedback
st.header("📈 Feedback")
feedback_file = Path("feedback_events.jsonl")
if feedback_file.exists():
    with open(feedback_file, 'r') as f:
        lines = f.readlines()
    st.write(f"Feedback events: {len(lines)}")
else:
    st.write("No feedback data found")
