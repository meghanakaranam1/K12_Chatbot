#!/usr/bin/env python3
"""
Flexible backend server starter - finds an available port automatically
No hardcoded ports, completely dynamic!
"""

import socket
import subprocess
import sys
import os
from pathlib import Path

def find_free_port(start_port=8787, max_attempts=20):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts}")

def start_server():
    """Start the backend server on an available port"""
    try:
        # Find available port
        port = find_free_port()
        print(f"🚀 Starting backend server on port {port}")
        
        # Get project root
        project_root = Path(__file__).parent
        
        # Use current Python (should be from venv if activated)
        python = sys.executable
        
        # Set PYTHONPATH to include project root so src module can be found
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        if current_pythonpath:
            env["PYTHONPATH"] = f"{project_root}:{current_pythonpath}"
        else:
            env["PYTHONPATH"] = str(project_root)
        
        # Start the server
        cmd = [
            python, "-m", "uvicorn", 
            "src.api.main:app", 
            "--reload", 
            "--port", str(port)
        ]
        
        print(f"📡 Running: {' '.join(cmd)}")
        print(f"🌐 Server will be available at: http://127.0.0.1:{port}")
        print(f"📚 API docs at: http://127.0.0.1:{port}/docs")
        print("\nPress Ctrl+C to stop the server")
        
        subprocess.run(cmd, cwd=project_root, env=env)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_server()
