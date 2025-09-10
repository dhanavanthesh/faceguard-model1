#!/usr/bin/env python3
"""
Standalone Sign Detection API - MediaPipe-based Gesture Recognition
Simple FastAPI server for gesture detection without face recognition dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Import sign detection
from sign_detection import SimplifiedSignDetector, initialize_sign_detector, get_sign_detector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Sign Detection API",
    description="MediaPipe-based gesture recognition API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
sign_detector_ready = False

# Pydantic models
class SignDetectionRequest(BaseModel):
    image_data: str

class SignDetectionResponse(BaseModel):
    success: bool
    gesture: str
    confidence: float
    is_emergency: bool
    hands_detected: int
    landmarks: Optional[Dict] = None
    timestamp: str
    message: Optional[str] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize sign detection on startup"""
    global sign_detector_ready
    logger.info("Starting Sign Detection API...")
    
    # Initialize sign detector
    if initialize_sign_detector():
        sign_detector_ready = True
        logger.info("Sign detection initialized successfully!")
    else:
        logger.warning("Sign detection initialization failed")
        sign_detector_ready = False

def process_base64_image(image_data: str) -> np.ndarray:
    """Convert base64 image to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image then to numpy array
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array (RGB format)
        image_array = np.array(pil_image)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sign Detection API",
        "status": "running" if sign_detector_ready else "initializing",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if sign_detector_ready else "initializing",
        "timestamp": datetime.now().isoformat(),
        "sign_detector_ready": sign_detector_ready
    }

@app.get("/api/signs/status")
async def get_sign_detection_status():
    """Get sign detection system status"""
    return {
        "success": True,
        "status": "running" if sign_detector_ready else "not_initialized",
        "mediapipe_loaded": sign_detector_ready,
        "models_available": True,
        "gesture_detection_ready": sign_detector_ready
    }

def convert_landmarks_to_dict(landmarks_data):
    """Convert MediaPipe landmarks to serializable dictionaries"""
    try:
        if not landmarks_data:
            return {'hands': [], 'pose': None}
        
        result = {'hands': [], 'pose': None}
        
        # Convert hand landmarks
        if 'hands' in landmarks_data:
            for hand_info in landmarks_data['hands']:
                hand_dict = {
                    'handedness': hand_info['handedness'],
                    'confidence': hand_info['confidence'],
                    'landmarks': []
                }
                
                # Convert landmarks to simple x,y coordinates
                if 'landmarks' in hand_info and hand_info['landmarks']:
                    for landmark in hand_info['landmarks'].landmark:
                        hand_dict['landmarks'].append({
                            'x': float(landmark.x),
                            'y': float(landmark.y),
                            'z': float(landmark.z) if hasattr(landmark, 'z') else 0.0
                        })
                
                result['hands'].append(hand_dict)
        
        # Convert pose landmarks (if any)
        if 'pose' in landmarks_data and landmarks_data['pose']:
            pose_landmarks = []
            for landmark in landmarks_data['pose'].landmark:
                pose_landmarks.append({
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z) if hasattr(landmark, 'z') else 0.0
                })
            result['pose'] = pose_landmarks
        
        return result
        
    except Exception as e:
        logger.error(f"Error converting landmarks: {e}")
        return {'hands': [], 'pose': None}

@app.post("/api/signs/detect", response_model=SignDetectionResponse)
async def detect_signs(request: SignDetectionRequest):
    """Detect signs/gestures in uploaded image"""
    if not sign_detector_ready:
        raise HTTPException(status_code=503, detail="Sign detection not initialized")
        
    try:
        sign_detector = get_sign_detector()
        if sign_detector is None:
            raise HTTPException(status_code=503, detail="Sign detector not available")
        
        # Process base64 image
        image_rgb = process_base64_image(request.image_data)
        
        # Convert RGB to BGR for OpenCV processing
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Use the sign detector
        result = sign_detector.process_frame(image_bgr)
        
        # Convert landmarks to serializable format
        landmarks = convert_landmarks_to_dict(result.get('landmarks', {}))
        
        return SignDetectionResponse(
            success=True,
            gesture=result.get('gesture', 'NONE'),
            confidence=result.get('confidence', 0.0),
            is_emergency=result.get('is_emergency', False),
            hands_detected=result.get('hands_detected', 0),
            landmarks=landmarks,
            timestamp=result.get('timestamp', datetime.now().isoformat()),
            message="Detection completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in sign detection: {e}")
        raise HTTPException(status_code=500, detail=f"Sign detection error: {str(e)}")

@app.get("/api/signs/logs")
async def get_sign_detection_logs(limit: int = 100):
    """Get recent sign detection logs"""
    try:
        sign_detector = get_sign_detector()
        if sign_detector is None:
            return {"success": False, "message": "Sign detector not available"}
        
        logs = sign_detector.get_detection_logs(limit)
        return {
            "success": True,
            "logs": logs,
            "count": len(logs)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")

@app.get("/api/signs/emergency-logs")
async def get_emergency_sign_logs(limit: int = 50):
    """Get recent emergency gesture logs"""
    try:
        sign_detector = get_sign_detector()
        if sign_detector is None:
            return {"success": False, "message": "Sign detector not available"}
        
        logs = sign_detector.get_emergency_logs(limit)
        return {
            "success": True,
            "logs": logs,
            "count": len(logs)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving emergency logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving emergency logs: {str(e)}")

@app.get("/api/signs/statistics")
async def get_sign_statistics():
    """Get sign detection statistics"""
    try:
        sign_detector = get_sign_detector()
        if sign_detector is None:
            return {"success": False, "message": "Sign detector not available"}
        
        stats = sign_detector.get_statistics()
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

@app.get("/api/signs/gestures")
async def get_supported_gestures():
    """Get list of supported gestures"""
    return {
        "success": True,
        "gestures": [
            {
                "name": "THUMBS_UP",
                "description": "Thumb extended upward",
                "emergency": False,
                "instructions": "Make a fist with thumb pointing up"
            },
            {
                "name": "PALM_STOP", 
                "description": "Open palm facing camera (stop signal)",
                "emergency": False,
                "instructions": "Show open palm with all fingers extended facing the camera"
            },
            {
                "name": "ARMS_CROSSED",
                "description": "Both arms crossed over chest (Emergency Signal)",
                "emergency": True,
                "instructions": "Cross both arms over your chest - this triggers emergency alerts"
            }
        ],
        "note": "Simplified system with 3 core gestures for better reliability"
    }

@app.delete("/api/signs/logs")
async def clear_sign_detection_logs():
    """Clear all sign detection logs"""
    try:
        sign_detector = get_sign_detector()
        if sign_detector is None:
            return {"success": False, "message": "Sign detector not available"}
        
        sign_detector.clear_logs()
        return {
            "success": True,
            "message": "Detection logs cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing logs: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)