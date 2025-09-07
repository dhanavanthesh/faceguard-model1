#!/usr/bin/env python3
"""
Virtual Environment Setup Script
"""

import os
import sys
import subprocess
import platform

def create_venv():
    """Create virtual environment"""
    print("🔧 Creating virtual environment...")
    
    venv_name = "face_recognition_env"
    
    try:
        # Create virtual environment
        subprocess.check_call([sys.executable, "-m", "venv", venv_name])
        print(f"✅ Virtual environment '{venv_name}' created successfully!")
        
        # Get activation command based on OS
        if platform.system() == "Windows":
            activate_cmd = f"{venv_name}\\Scripts\\activate"
            pip_path = f"{venv_name}\\Scripts\\pip"
        else:
            activate_cmd = f"source {venv_name}/bin/activate"
            pip_path = f"{venv_name}/bin/pip"
            
        print(f"\n📋 To activate virtual environment:")
        print(f"   {activate_cmd}")
        print(f"\n📦 To install requirements:")
        print(f"   {pip_path} install -r requirements.txt")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def install_in_venv():
    """Install requirements in virtual environment"""
    venv_name = "face_recognition_env"
    
    if platform.system() == "Windows":
        pip_path = f"{venv_name}\\Scripts\\pip"
        python_path = f"{venv_name}\\Scripts\\python"
    else:
        pip_path = f"{venv_name}/bin/pip"
        python_path = f"{venv_name}/bin/python"
        
    if not os.path.exists(pip_path):
        print("❌ Virtual environment not found. Run create_venv() first.")
        return False
        
    try:
        print("📦 Installing requirements in virtual environment...")
        subprocess.check_call([pip_path, "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        
        print(f"\n🚀 To run the application:")
        print(f"   {python_path} main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "create":
            create_venv()
        elif sys.argv[1] == "install":
            install_in_venv()
        elif sys.argv[1] == "setup":
            if create_venv():
                install_in_venv()
    else:
        print("🐍 Virtual Environment Setup")
        print("Usage:")
        print("  python setup_venv.py create  - Create virtual environment")
        print("  python setup_venv.py install - Install requirements in venv")
        print("  python setup_venv.py setup   - Create venv and install requirements")