#!/usr/bin/env python3
"""
Run script for the Face Recognition Backend
"""

import sys
import os
import subprocess

def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def run_training():
    """Run the training module"""
    try:
        print("🎯 Starting training...")
        from training import main as training_main
        training_main()
        print("✅ Training completed!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def run_server():
    """Run the FastAPI server"""
    try:
        print("🚀 Starting FastAPI server...")
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "install":
            install_requirements()
        elif command == "train":
            run_training()
        elif command == "server":
            run_server()
        else:
            print("Usage:")
            print("  python run.py install  - Install requirements")
            print("  python run.py train    - Run training")
            print("  python run.py server   - Start FastAPI server")
    else:
        print("🎯 Face Recognition Backend")
        print("\nUsage:")
        print("  python run.py install  - Install requirements")
        print("  python run.py train    - Run training")
        print("  python run.py server   - Start FastAPI server")
        print("\nQuick start:")
        print("  1. python run.py install")
        print("  2. python run.py train")
        print("  3. python run.py server")