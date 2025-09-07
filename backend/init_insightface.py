#!/usr/bin/env python3
"""
Simple InsightFace initialization script
"""

def test_insightface():
    """Test InsightFace initialization"""
    print("🧪 Testing InsightFace initialization...")
    
    try:
        import insightface
        from insightface.app import FaceAnalysis
        print("✅ InsightFace imported successfully")
        
        print("📥 Initializing FaceAnalysis with buffalo_l...")
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        
        print("🔧 Preparing model...")
        app.prepare(ctx_id=-1, det_size=(640, 640))  # Use -1 for CPU
        
        print("✅ InsightFace initialized successfully!")
        print(f"📊 Available providers: {app.providers}")
        
        # Test with a small dummy image
        import numpy as np
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = app.get(dummy_image)
        print(f"🎯 Test detection result: {len(faces)} faces found (expected: 0 for black image)")
        
        return True
        
    except Exception as e:
        print(f"❌ InsightFace initialization failed: {e}")
        
        # Check common issues
        try:
            import onnxruntime
            print(f"✅ ONNXRuntime version: {onnxruntime.__version__}")
        except ImportError:
            print("❌ ONNXRuntime not installed")
            
        try:
            import cv2
            print(f"✅ OpenCV version: {cv2.__version__}")
        except ImportError:
            print("❌ OpenCV not installed")
            
        return False

if __name__ == "__main__":
    success = test_insightface()
    if success:
        print("🎉 InsightFace is ready to use!")
    else:
        print("⚠️ InsightFace needs attention")