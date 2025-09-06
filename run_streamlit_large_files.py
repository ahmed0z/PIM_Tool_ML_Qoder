#!/usr/bin/env python3
"""
Run AutoPatternChecker Streamlit App with Large File Support
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Main function."""
    print("🚀 Starting AutoPatternChecker with Large File Support...")
    
    # Check if we're in the right directory
    if not os.path.exists("streamlit_app.py"):
        print("❌ streamlit_app.py not found!")
        print("Please run this script from the AutoPatternChecker directory")
        return 1
    
    # Set environment variables for large file support
    env = os.environ.copy()
    env['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '200'
    env['STREAMLIT_SERVER_MAX_MESSAGE_SIZE'] = '200'
    
    # Start Streamlit with large file support
    try:
        print("📁 Large file upload limit: 200MB")
        print("🌐 Starting Streamlit app...")
        print("🔗 Open your browser to: http://localhost:8501")
        print("⚠️  For very large files (>100MB), consider using handle_large_files.py first")
        print("\n" + "="*50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false",
            "--server.maxUploadSize", "200"
        ], env=env)
        
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return 1

if __name__ == "__main__":
    exit(main())