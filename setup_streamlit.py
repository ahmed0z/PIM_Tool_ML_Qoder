#!/usr/bin/env python3
"""
Setup script for AutoPatternChecker Streamlit App
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error {description}: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Please use Python 3.8 or higher")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor} is compatible")
        return True

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    
    # Check if requirements file exists
    if not Path("requirements_streamlit.txt").exists():
        print("❌ requirements_streamlit.txt not found!")
        return False
    
    # Install requirements
    return run_command(
        f"{sys.executable} -m pip install -r requirements_streamlit.txt",
        "Installing Python packages"
    )

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "output", 
        "indices",
        "trained_models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created {directory}/")
    
    return True

def create_sample_data():
    """Create sample data for testing."""
    print("📊 Creating sample data...")
    
    try:
        from demo_streamlit import create_sample_data
        sample_data = create_sample_data()
        sample_data.to_csv("data/sample_data.csv", index=False)
        print(f"   ✅ Created data/sample_data.csv ({len(sample_data)} rows)")
        return True
    except Exception as e:
        print(f"   ⚠️  Could not create sample data: {e}")
        return False

def test_imports():
    """Test if all imports work."""
    print("🧪 Testing imports...")
    
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        from autopatternchecker import DataIngester, PatternProfiler, ModelTrainer
        print("   ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

def create_startup_script():
    """Create a startup script."""
    print("📝 Creating startup script...")
    
    startup_script = """#!/usr/bin/env python3
\"\"\"
AutoPatternChecker Streamlit Startup Script
\"\"\"

import subprocess
import sys
import os

def main():
    print("🚀 Starting AutoPatternChecker Streamlit App...")
    
    # Check if we're in the right directory
    if not os.path.exists("streamlit_app.py"):
        print("❌ streamlit_app.py not found!")
        print("Please run this script from the AutoPatternChecker directory")
        return
    
    # Start Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\\n👋 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

if __name__ == "__main__":
    main()
"""
    
    with open("start_streamlit.py", "w") as f:
        f.write(startup_script)
    
    # Make it executable
    os.chmod("start_streamlit.py", 0o755)
    print("   ✅ Created start_streamlit.py")
    return True

def show_next_steps():
    """Show next steps to the user."""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the Streamlit app:")
    print("   python start_streamlit.py")
    print("   OR")
    print("   streamlit run streamlit_app.py")
    print("\n2. Open your browser to:")
    print("   http://localhost:8501")
    print("\n3. Upload sample data:")
    print("   - Go to 'Data Upload' page")
    print("   - Upload data/sample_data.csv")
    print("   - Configure columns and process data")
    print("\n4. Explore the app:")
    print("   - Pattern Analysis: See data patterns")
    print("   - Model Training: Train ML models")
    print("   - Validation: Test value validation")
    print("   - Analytics: View data insights")
    print("\n🎯 Happy exploring!")

def main():
    """Main setup function."""
    print("🔍 AutoPatternChecker Streamlit Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    if not create_directories():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Test imports
    if not test_imports():
        print("❌ Setup failed - import errors")
        return
    
    # Create sample data
    create_sample_data()
    
    # Create startup script
    create_startup_script()
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()