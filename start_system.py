#!/usr/bin/env python3
"""
Quick Start Script for Face Recognition System
"""

import os
import sys
import subprocess
import platform
import time

def check_node():
    """Check if Node.js is available"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    print("❌ Node.js not found. Please install Node.js first.")
    return False

def check_python():
    """Check Python version"""
    version = sys.version_info
    print(f"✅ Python: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        return True
    print("❌ Python 3.8+ required")
    return False

def setup_backend():
    """Setup backend with virtual environment"""
    print("\n🔧 Setting up Backend...")
    
    os.chdir('backend')
    
    # Create virtual environment and install
    try:
        subprocess.check_call([sys.executable, 'setup_venv.py', 'setup'])
        print("✅ Backend setup complete!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Backend setup failed")
        return False
    finally:
        os.chdir('..')

def setup_frontend():
    """Setup frontend"""
    print("\n📦 Setting up Frontend...")
    
    os.chdir('frontend')
    
    try:
        # Install npm packages
        subprocess.check_call(['npm', 'install'])
        print("✅ Frontend setup complete!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Frontend setup failed")
        return False
    finally:
        os.chdir('..')

def run_training():
    """Run training"""
    print("\n🎯 Running Training...")
    
    os.chdir('backend')
    
    try:
        venv_name = "face_recognition_env"
        
        if platform.system() == "Windows":
            python_path = f"{venv_name}\\Scripts\\python"
        else:
            python_path = f"{venv_name}/bin/python"
            
        if os.path.exists(python_path):
            subprocess.check_call([python_path, 'training.py'])
        else:
            subprocess.check_call([sys.executable, 'training.py'])
            
        print("✅ Training complete!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Training failed")
        return False
    finally:
        os.chdir('..')

def main():
    """Main setup function"""
    print("🚀 Face Recognition System Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python():
        return
    if not check_node():
        return
        
    # Setup components
    if not setup_backend():
        return
    if not setup_frontend():
        return
        
    # Ask about training
    train = input("\n🎯 Run training now? (y/n): ").lower().startswith('y')
    if train:
        run_training()
        
    print("\n" + "=" * 50)
    print("✅ Setup Complete!")
    print("\n🚀 To start the system:")
    print("Terminal 1: cd backend && python main.py")
    print("Terminal 2: cd frontend && npm start")
    print("\n🌐 Then open: http://localhost:3000")

if __name__ == "__main__":
    main()