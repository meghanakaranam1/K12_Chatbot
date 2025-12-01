#!/usr/bin/env python3
"""
Setup script for ClassroomGPT
"""
import os
import subprocess
import sys
from pathlib import Path

def main():
    print("🎓 ClassroomGPT Setup")
    print("=" * 50)
    
    # Check if requirements are installed
    print("📦 Checking dependencies...")
    try:
        import streamlit
        import pandas
        import numpy
        import faiss
        import sentence_transformers
        import openai
        print("✅ All dependencies are installed!")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Installing requirements...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run: pip install -r requirements.txt")
            return
    
    # Check for secrets file
    secrets_file = Path(".streamlit/secrets.toml")
    if secrets_file.exists():
        print("✅ Secrets file found")
    else:
        print("⚠️  No secrets file found. Creating sample...")
        secrets_file.parent.mkdir(exist_ok=True)
        with open(secrets_file, 'w') as f:
            f.write("""# ClassroomGPT Secrets Configuration
# Add your actual API keys here

# OpenAI Configuration (Required for AI features)
OPENAI_API_KEY = "your-openai-api-key-here"

# Supabase Configuration (Optional - for PDF storage)
SUPABASE_URL = "your-supabase-url-here"
SUPABASE_SERVICE_ROLE_KEY = "your-supabase-service-role-key-here"
SUPABASE_BUCKET = "activities"
SIGNED_URL_TTL = "3600"
""")
        print("📝 Sample secrets file created at .streamlit/secrets.toml")
        print("   Please edit it with your actual API keys!")
    
    # Check for data file
    data_file = Path("data/activities.csv")
    if data_file.exists():
        print("✅ Data file found")
    else:
        print("⚠️  No data file found. Using sample data.")
    
    print("\n🚀 Setup complete!")
    print("\nTo run the chatbot:")
    print("  python run.py")
    print("\nTo run the admin dashboard:")
    print("  python run.py admin")
    print("\nDon't forget to add your OpenAI API key to .streamlit/secrets.toml!")

if __name__ == "__main__":
    main()
