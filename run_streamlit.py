#!/usr/bin/env python3
"""
Run AutoPatternChecker Streamlit App
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def run_streamlit():
    """Run the Streamlit app."""
    print("🚀 Starting AutoPatternChecker Streamlit App...")
    
    # Check if streamlit_app.py exists
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found!")
        return False
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])
        return True
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        return False

def main():
    """Main function."""
    print("🔍 AutoPatternChecker Streamlit Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("autopatternchecker").exists():
        print("❌ AutoPatternChecker package not found!")
        print("Please run this script from the AutoPatternChecker root directory")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Run streamlit
    run_streamlit()

if __name__ == "__main__":
    main()