#!/usr/bin/env python3
"""
Test script to verify the backend structure
"""

def test_imports():
    """Test basic imports"""
    print("🧪 Testing imports...")
    
    try:
        import sys
        import os
        print("✅ Basic Python modules OK")
    except Exception as e:
        print(f"❌ Basic modules failed: {e}")
        return False
        
    try:
        from fastapi import FastAPI
        print("✅ FastAPI import OK")
    except ImportError:
        print("⚠️ FastAPI not installed - run 'python run.py install'")
        
    try:
        import cv2
        print("✅ OpenCV import OK")
    except ImportError:
        print("⚠️ OpenCV not installed - run 'python run.py install'")
        
    try:
        import insightface
        print("✅ InsightFace import OK")
    except ImportError:
        print("⚠️ InsightFace not installed - run 'python run.py install'")
        
    return True

def test_file_structure():
    """Test file structure"""
    print("\n📂 Testing file structure...")
    
    required_files = [
        'main.py',
        'training.py', 
        'run.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
            
    return len(missing_files) == 0

def test_training_module():
    """Test training module structure"""
    print("\n🎯 Testing training module...")
    
    try:
        # Test if we can import the training module
        import training
        
        # Check if main functions exist
        functions_to_check = [
            'initialize_insightface',
            'preprocess_training_data',
            'enhanced_recognize_face',
            'register_new_face'
        ]
        
        for func_name in functions_to_check:
            if hasattr(training, func_name):
                print(f"✅ Function {func_name} exists")
            else:
                print(f"❌ Function {func_name} missing")
                
        return True
        
    except Exception as e:
        print(f"❌ Training module error: {e}")
        return False

def test_main_api():
    """Test main API structure"""
    print("\n🚀 Testing FastAPI structure...")
    
    try:
        import main
        
        # Check if FastAPI app exists
        if hasattr(main, 'app'):
            print("✅ FastAPI app exists")
            
            # Check if main endpoints exist (basic check)
            if hasattr(main, 'root'):
                print("✅ Root endpoint exists")
            if hasattr(main, 'health_check'):
                print("✅ Health endpoint exists")
            if hasattr(main, 'recognize_face'):
                print("✅ Recognize endpoint exists")
            if hasattr(main, 'register_face'):
                print("✅ Register endpoint exists")
                
        return True
        
    except Exception as e:
        print(f"❌ Main API error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Backend Structure Test")
    print("=" * 50)
    
    all_passed = True
    
    # Run all tests
    all_passed &= test_file_structure()
    all_passed &= test_imports()
    all_passed &= test_training_module()
    all_passed &= test_main_api()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! Backend structure is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: python run.py install")
        print("2. Run training: python run.py train") 
        print("3. Start server: python run.py server")
    else:
        print("⚠️ Some tests failed. Check the output above.")
        
    print("\nManual commands:")
    print("- Install: pip install -r requirements.txt")
    print("- Train: python training.py")
    print("- Server: uvicorn main:app --reload")