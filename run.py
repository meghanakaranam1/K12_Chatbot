#!/usr/bin/env python3
"""
Simple launcher script for ClassroomGPT
"""
import subprocess
import sys
import os

def main():
    """Launch the appropriate app"""
    print("🔍 [DEBUG] run.py: main() called")
    if len(sys.argv) > 1:
        print(f"🔍 [DEBUG] run.py: Command line args: {sys.argv[1:]}")
        if sys.argv[1] == "admin":
            app_path = "apps/admin.py"
            print("🚀 Starting ClassroomGPT Admin Dashboard...")
        elif sys.argv[1] == "chatgpt":
            app_path = "apps/chatgpt_like.py"
            print("🚀 Starting ClassroomGPT - Intelligent AI Mode...")
        else:
            app_path = "apps/main.py"
            print("🚀 Starting ClassroomGPT (Simple Mode)...")
    else:
        print("🎓 ClassroomGPT Launcher")
        print("=" * 40)
        print("Choose your mode:")
        print("1. Simple Mode (original interface)")
        print("2. Intelligent AI Mode (full AI assistant)")
        print("3. Admin Dashboard")
        print()
        
        choice = input("Enter your choice (1-3): ").strip()
        print(f"🔍 [DEBUG] run.py: User choice: {choice}")
        
        if choice == "1":
            app_path = "apps/main.py"
            print("🚀 Starting ClassroomGPT (Simple Mode)...")
        elif choice == "2":
            app_path = "apps/chatgpt_like.py"
            print("🚀 Starting ClassroomGPT - Intelligent AI Mode...")
        elif choice == "3":
            app_path = "apps/admin.py"
            print("🚀 Starting ClassroomGPT Admin Dashboard...")
        else:
            app_path = "apps/main.py"
            print("🚀 Starting ClassroomGPT (Simple Mode)...")
    
    print(f"🔍 [DEBUG] run.py: Launching app at: {app_path}")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
