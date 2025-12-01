"""
Configuration management for ClassroomGPT
"""
import os
import streamlit as st
from pathlib import Path
from typing import Optional

class Config:
    """Centralized configuration management"""
    
    def __init__(self):
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables and Streamlit secrets"""
        # OpenAI Configuration
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.openai_api_key = self._get_secret_or_env("OPENAI_API_KEY")
        
        # Embedding Configuration
        self.embed_model = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        
        # Data Configuration
        self.data_path = os.getenv(
            "DATA_PATH", 
            "data/activities.csv"
        )
        
        # Search Configuration
        self.top_k = int(os.getenv("TOP_K", "8"))
        self.max_results = int(os.getenv("MAX_RESULTS", "8"))
        
        # Supabase Configuration
        self.supabase_url = self._get_secret_or_env("SUPABASE_URL", "")
        self.supabase_key = self._get_secret_or_env("SUPABASE_SERVICE_ROLE_KEY", 
                                                   os.getenv("SUPABASE_ANON_KEY", ""))
        self.supabase_bucket = self._get_secret_or_env("SUPABASE_BUCKET", "activities")
        self.signed_url_ttl = int(self._get_secret_or_env("SIGNED_URL_TTL", "3600"))
        
        # File paths
        self.sessions_file = Path("sessions.json")
        
        # Search columns for data processing
        self.search_columns = [
            "Strategic Action", "Short Description", "Notes", "Category",
            "Time to prepare/learn", "Time to implement", "Time",
            "Objective", "Advance preparation", "Materials", "Student Materials",
            "Introduction", "Additional Resources", "Link to the Source Materials",
            "Source Material Type (Video/Web Article/Journal Article, etc.)",
            "Storage Key", "Storage URL", "Area of School Connectedness",
            "Link to Resource", "Reference Link", "Source Material Type",
            "Implementation Level (Classroom (C)/School (S) /Both(B))",
            "Activity Type (Activity (A) or Professional Learning (PL))",
            "Ease of Use", "Resources Needed", "Building Block"
        ]
    
    @property
    def is_supabase_configured(self) -> bool:
        """Check if Supabase is properly configured"""
        return bool(self.supabase_url and self.supabase_key)
    
    @property
    def is_openai_configured(self) -> bool:
        """Check if OpenAI is properly configured"""
        return bool(self.openai_api_key)
    
    def _get_secret_or_env(self, key: str, default: str = None) -> str:
        """Safely get value from Streamlit secrets or environment variables"""
        try:
            return st.secrets.get(key, os.getenv(key, default))
        except Exception:
            # If secrets file doesn't exist, fall back to environment variables
            return os.getenv(key, default)

# Global config instance
config = Config()
