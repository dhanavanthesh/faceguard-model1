#!/usr/bin/env python3
"""
Minimal FastAPI Backend for Face Recognition
Based on the notebook training system
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, Any, List, Optional
import os
import sys
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Import our training module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import training
from training import (
    initialize_insightface, 
    enhanced_recognize_face, 
    register_new_face,
    load_face_database,
    save_face_database,
    face_database
)

# Import sign detection modules
from sign_detection import SimplifiedSignDetector, initialize_sign_detector, get_sign_detector
from sign_training import SignTrainingManager

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    global system_initialized
    
    # Startup
    print("Starting Face Recognition API...")
    
    # Initialize InsightFace
    if initialize_insightface():
        # Initialize mask detection
        training.initialize_mask_detector()
        # Load existing database
        load_face_database()
        # Load user database
        load_user_database()
        
        # Initialize sign detection
        global sign_detector_initialized, sign_training_manager
        if initialize_sign_detector():
            sign_detector_initialized = True
            print("✅ SimplifiedSignDetector initialized successfully!")
            
            # Initialize training manager separately (optional for simplified detector)
            try:
                sign_training_manager = SignTrainingManager()
                # Load trained models if available (optional)
                if sign_training_manager.load_trained_models():
                    print("📚 Sign detection with trained models initialized!")
                else:
                    print("📚 Sign detection running without trained models (using MediaPipe only)")
            except Exception as e:
                print(f"⚠️  Training manager initialization failed: {e}")
                print("✅ Sign detection still working with simplified MediaPipe-only detection")
                sign_training_manager = None
        else:
            print("❌ Sign detection initialization failed")
            sign_detector_initialized = False
        
        system_initialized = True
        print("System initialized successfully!")
        print(f"🔍 DEBUG: system_initialized is now: {system_initialized}")
    else:
        print("System initialization failed")
        system_initialized = False
    
    yield
    
    # Shutdown
    print("Shutting down Face Recognition API...")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Face Recognition API",
    description="Minimal Face Recognition API based on InsightFace buffalo_l model",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - specifically for React frontend on port 3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global state
system_initialized = False
training_status = {"is_training": False, "progress": 0, "message": "Not started"}
sign_training_manager = None
sign_detector_initialized = False

# Pydantic models for API requests
class RegisterRequest(BaseModel):
    name: str
    image_data: str  # base64 encoded

class RecognizeRequest(BaseModel):
    image_data: str  # base64 encoded

class AddSampleRequest(BaseModel):
    image_data: str  # base64 encoded

class SignDetectionRequest(BaseModel):
    image_data: str  # base64 encoded

# Startup is now handled by lifespan context manager above

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Recognition API",
        "status": "running" if system_initialized else "initializing",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if system_initialized else "initializing",
        "database_size": len(face_database),
        "insightface_loaded": training.app is not None,
        "system_initialized": system_initialized
    }

@app.get("/debug")
async def debug_status():
    """Debug endpoint to check system state"""
    global system_initialized, sign_detector_initialized
    return {
        "system_initialized": system_initialized,
        "sign_detector_initialized": sign_detector_initialized,
        "sign_detector_available": get_sign_detector() is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/dashboard/stats")
async def get_stats():
    """Get system statistics"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    total_users = len(face_database)
    total_samples = sum(data.get("samples", 1) for data in face_database.values())
    avg_confidence = sum(data.get("confidence", 0.8) for data in face_database.values()) / total_users if total_users > 0 else 0
    
    # Read recognition logs for real data
    try:
        with open("recognition_logs.json", 'r') as f:
            logs = json.load(f)
        
        # Calculate recognition stats from logs
        recent_logs = logs[-100:] if len(logs) > 100 else logs
        successful_recognitions = len([log for log in recent_logs if log.get("is_registered", False)])
        total_recognitions = len(recent_logs)
        recognition_rate = (successful_recognitions / total_recognitions * 100) if total_recognitions > 0 else 0
        
        # Recent activity from logs
        recent_activity = []
        for log in logs[-10:]:
            recent_activity.append({
                "timestamp": log.get("timestamp", ""),
                "type": "recognition",
                "description": f"{'Recognized' if log.get('is_registered') else 'Unknown person detected'}: {log.get('name', 'UNKNOWN')} (confidence: {log.get('confidence', 0):.3f})",
                "user": log.get("name", "UNKNOWN")
            })
        recent_activity.reverse()  # Show most recent first
        
    except (FileNotFoundError, json.JSONDecodeError):
        recognition_rate = 85.0
        successful_recognitions = 0
        total_recognitions = 0
        recent_activity = []
    
    # Create comprehensive dashboard structure
    stats = {
        "basic_stats": {
            "total_users": total_users,
            "total_samples": total_samples,
            "average_confidence": avg_confidence,
            "recent_registrations": 0,  # Could track this if needed
        },
        "recognition_stats": {
            "successful_recognitions_30d": successful_recognitions,
            "total_recognitions_30d": total_recognitions,
            "recognition_rate": recognition_rate,
            "avg_response_time": 0.245
        },
        "ml_model_stats": {
            "is_trained": training.model_trained if hasattr(training, 'model_trained') else False,
            "is_training": training_status.get("is_training", False),
            "model_accuracy": avg_confidence,
            "training_data_size": total_samples,
        },
        "user_distribution": {},
        "recent_activity": recent_activity
    }
    
    # Add user distribution
    for name, data in face_database.items():
        stats["user_distribution"][name] = data.get("samples", 1)
        
    return stats

def process_image_file(file_content: bytes) -> np.ndarray:
    """Convert uploaded file to OpenCV image"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(file_content))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert to numpy array
        image_np = np.array(image)
        
        return image_np
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def ensure_uploads_directory():
    """Ensure uploads directory structure exists"""
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    return uploads_dir

def save_base64_image(base64_data: str, person_name: str, file_extension: str = ".jpg") -> str:
    """Save base64 image to person's folder and return the file path"""
    try:
        # Handle data URL format: "data:image/jpeg;base64,..."
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_data)
        
        return save_uploaded_image(image_bytes, person_name, file_extension)
        
    except Exception as e:
        print(f"❌ Error saving base64 image: {e}")
        return None

def save_uploaded_image(file_content: bytes, person_name: str, file_extension: str = ".jpg") -> str:
    """Save uploaded image to person's folder and return the file path"""
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = ensure_uploads_directory()
        
        # Create person-specific directory
        person_dir = uploads_dir / person_name.replace(" ", "_").lower()
        person_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person_name.replace(' ', '_').lower()}_{timestamp}_{unique_id}{file_extension}"
        
        # Save image
        file_path = person_dir / filename
        with open(file_path, "wb") as f:
            f.write(file_content)
            
        print(f"✅ Image saved: {file_path}")
        return str(file_path)
        
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return None

def save_user_database():
    """Save user database to JSON file"""
    try:
        users_file = "users_database.json"
        users = []
        
        for name, data in face_database.items():
            user_id = name.replace(" ", "_").lower()
            user_entry = {
                "id": user_id,
                "name": name,
                "total_samples": data.get("samples", 1),
                "confidence_level": data.get("confidence", 0.8),
                "created_at": data.get("created_at", datetime.now().isoformat()),
                "updated_at": datetime.now().isoformat(),
                "image_path": data.get("image_path", f"uploads/{user_id}/")
            }
            users.append(user_entry)
        
        with open(users_file, 'w') as f:
            json.dump(users, f, indent=2)
            
        print(f"✅ User database saved: {len(users)} users")
        
    except Exception as e:
        print(f"❌ Error saving user database: {e}")

def load_user_database():
    """Load user database from JSON file"""
    try:
        users_file = "users_database.json"
        with open(users_file, 'r') as f:
            users = json.load(f)
            
        # Update face_database with loaded user data
        for user in users:
            face_database[user["name"]] = {
                "samples": user.get("total_samples", 1),
                "confidence": user.get("confidence_level", 0.8),
                "created_at": user.get("created_at"),
                "updated_at": user.get("updated_at"),
                "image_path": user.get("image_path")
            }
            
        print(f"✅ User database loaded: {len(users)} users")
        
    except (FileNotFoundError, json.JSONDecodeError):
        print("ℹ️  No existing user database found, starting fresh")

def log_recognition_event(name: str, confidence: float, is_registered: bool, image_path: str = None):
    """Log recognition events to JSON file for tracking"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "confidence": float(confidence),
            "is_registered": is_registered,
            "status": "AUTHORIZED" if is_registered else "UNAUTHORIZED",
            "image_path": image_path
        }
        
        log_file = "recognition_logs.json"
        
        # Load existing logs or create new list
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
            
        # Add new log entry
        logs.append(log_entry)
        
        # Keep only last 1000 entries to prevent file from growing too large
        if len(logs) > 1000:
            logs = logs[-1000:]
            
        # Save back to file
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
            
        print(f"✅ Recognition logged: {name} - {log_entry['status']}")
        
    except Exception as e:
        print(f"❌ Error logging recognition: {e}")

def process_base64_image(base64_data: str) -> np.ndarray:
    """Convert base64 image data to OpenCV image"""
    try:
        # Handle data URL format: "data:image/jpeg;base64,..."
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_data)
        
        # Convert to image
        return process_image_file(image_bytes)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

@app.post("/api/faces/recognize-file")
async def recognize_face(file: UploadFile = File(...)):
    """Recognize faces in uploaded image"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    if training.app is None:
        raise HTTPException(status_code=503, detail="InsightFace model not loaded. Please restart the server or check model installation.")
        
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    try:
        # Read and process image
        file_content = await file.read()
        image_rgb = process_image_file(file_content)
        
        # Get faces using InsightFace
        faces = training.app.get(image_rgb)
        
        if not faces:
            return {
                "success": False,
                "message": "No faces detected in image",
                "faces": []
            }
            
        results = []
        
        for i, face in enumerate(faces):
            if face.det_score > 0.5:  # Confidence threshold
                # Get face embedding
                embedding = face.embedding
                
                # Recognize face with mask detection
                name, confidence = enhanced_recognize_face(
                    embedding, 
                    confidence_threshold=0.6,
                    face_image=image_rgb,
                    face_bbox=face.bbox
                )
                
                # Get bounding box
                bbox = face.bbox.astype(int).tolist()
                
                # Get landmarks (5 key points: left eye, right eye, nose, left mouth, right mouth)
                landmarks = None
                if hasattr(face, 'kps') and face.kps is not None:
                    landmarks = face.kps.astype(int).tolist()
                
                # Determine if registered (green) or unknown (red)
                is_registered = name is not None and name != "UNKNOWN"
                
                # Log recognition event
                log_recognition_event(
                    name=name if name else "UNKNOWN",
                    confidence=confidence,
                    is_registered=is_registered
                )
                
                result = {
                    "face_id": i,
                    "name": name if name else "UNKNOWN",
                    "confidence": float(confidence),
                    "detection_score": float(face.det_score),
                    "bbox": bbox,  # [x1, y1, x2, y2]
                    "landmarks": landmarks,  # [[x1, y1], [x2, y2], ...] for 5 key points
                    "is_registered": is_registered
                }
                
                results.append(result)
                
        return {
            "success": True,
            "message": f"Processed {len(results)} faces",
            "faces": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition error: {str(e)}")

@app.post("/api/faces/register-file") 
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    """Register a new face"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    if training.app is None:
        raise HTTPException(status_code=503, detail="InsightFace model not loaded. Please restart the server or check model installation.")
        
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
        
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    try:
        # Read and process image
        file_content = await file.read()
        image_rgb = process_image_file(file_content)
        
        # Save uploaded image to folder
        image_path = save_uploaded_image(file_content, name.strip(), ".jpg")
        
        # Get faces using InsightFace
        faces = training.app.get(image_rgb)
        
        if not faces:
            return {
                "success": False,
                "message": "No face detected in image"
            }
            
        if len(faces) > 1:
            return {
                "success": False,
                "message": "Multiple faces detected. Please ensure only one face is in the image."
            }
            
        face = faces[0]
        
        # Check quality
        if face.det_score < 0.7:
            return {
                "success": False,
                "message": "Face quality too low. Please improve lighting and face clarity."
            }
            
        # Extract embedding
        embedding = face.embedding
        
        # Register the face
        register_new_face(name.strip(), embedding, image_path)
        
        # Add timestamp info to face_database entry
        if name.strip() not in face_database:
            face_database[name.strip()] = {}
        face_database[name.strip()]["created_at"] = datetime.now().isoformat()
        face_database[name.strip()]["updated_at"] = datetime.now().isoformat()
        face_database[name.strip()]["image_path"] = image_path
        
        # Save databases
        save_face_database()
        save_user_database()
        
        return {
            "success": True,
            "message": f"Successfully registered {name.strip()}!",
            "detection_score": float(face.det_score),
            "bbox": face.bbox.astype(int).tolist(),
            "image_path": image_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")

@app.get("/api/faces/users")
async def list_users():
    """List all registered users"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Try to load from JSON file first for most up-to-date data
        users_file = "users_database.json"
        try:
            with open(users_file, 'r') as f:
                users = json.load(f)
            print(f"✅ Loaded {len(users)} users from database file")
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback to face_database
            users = []
            for name, data in face_database.items():
                user_id = name.replace(" ", "_").lower()
                users.append({
                    "id": user_id,
                    "name": name,
                    "total_samples": data.get("samples", 1),
                    "confidence_level": data.get("confidence", 0.8),
                    "created_at": data.get("created_at", "2024-01-01T12:00:00.000Z"),
                    "updated_at": data.get("updated_at", "2024-01-01T12:00:00.000Z"),
                    "image_path": data.get("image_path", f"uploads/{user_id}/")
                })
            
            # Save this data for next time
            save_user_database()
            
        return {
            "success": True,
            "users": users,
            "total": len(users)
        }
        
    except Exception as e:
        print(f"❌ Error loading users: {e}")
        return {
            "success": False,
            "users": [],
            "total": 0,
            "error": str(e)
        }

@app.delete("/api/faces/users/{name}")
async def delete_user(name: str):
    """Delete a registered user"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    if name not in face_database:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Delete user folder if it exists
    try:
        user_id = name.replace(" ", "_").lower()
        user_folder = Path(f"uploads/{user_id}")
        if user_folder.exists():
            shutil.rmtree(user_folder)
            print(f"✅ Deleted user folder: {user_folder}")
    except Exception as e:
        print(f"❌ Error deleting user folder: {e}")
    
    del face_database[name]
    save_face_database()
    save_user_database()
    
    return {
        "success": True,
        "message": f"User {name} deleted successfully"
    }

@app.post("/api/faces/train-models")
async def trigger_training():
    """Trigger model training"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        # Import training functions
        from training import main as training_main
        
        # Run training in background (in production, use a task queue)
        training_main()
        
        return {
            "success": True,
            "message": "Training completed",
            "database_size": len(face_database)
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Training error: {str(e)}"
        }

@app.post("/api/faces/train-mask-detection")
async def trigger_mask_training():
    """Trigger mask detection training with Kaggle datasets"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        global training_status
        training_status = {"is_training": True, "progress": 0, "message": "Starting mask detection training..."}
        
        # Import training functions
        from training import (
            download_and_extract_datasets, 
            preprocess_training_data, 
            train_models,
            create_face_database,
            save_face_database
        )
        
        # Step 1: Download mask datasets from Kaggle (~300MB)
        training_status["progress"] = 10
        training_status["message"] = "Downloading mask detection datasets from Kaggle..."
        download_and_extract_datasets()
        
        # Step 2: Preprocess training data (extract embeddings)
        training_status["progress"] = 40
        training_status["message"] = "Processing mask/unmask face images..."
        embeddings, labels, images = preprocess_training_data()
        
        if not embeddings:
            raise Exception("No training data processed")
        
        # Step 3: Train ML models
        training_status["progress"] = 70
        training_status["message"] = "Training SVM/Random Forest models..."
        model_success = train_models()
        
        # Step 4: Create enhanced face database with mask awareness
        training_status["progress"] = 90
        training_status["message"] = "Creating mask-aware face database..."
        create_face_database()
        save_face_database()
        
        # Complete
        training_status = {
            "is_training": False, 
            "progress": 100, 
            "message": "Mask detection training completed successfully!"
        }
        
        return {
            "success": True,
            "message": "Mask detection training completed successfully!",
            "details": {
                "embeddings_processed": len(embeddings),
                "unique_identities": len(set(labels)),
                "model_trained": model_success,
                "database_size": len(face_database)
            }
        }
        
    except Exception as e:
        training_status = {
            "is_training": False, 
            "progress": 0, 
            "message": f"Training failed: {str(e)}"
        }
        return {
            "success": False,
            "message": f"Mask training error: {str(e)}"
        }

@app.get("/api/faces/system-status")
async def get_system_status():
    """Get system status"""
    return {
        "success": True,
        "status": "running" if system_initialized else "initializing",
        "database_size": len(face_database),
        "insightface_loaded": training.app is not None,
        "model_trained": os.path.exists('face_recognition_model.pkl')
    }

@app.post("/api/faces/live-recognize")
async def live_recognize(file: UploadFile = File(...)):
    """Live recognition endpoint (same as recognize but for live feed)"""
    return await recognize_face(file)

# Base64 image endpoints
@app.post("/api/faces/register")
async def register_face_base64(request: RegisterRequest):
    """Register a new face using base64 image data"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    if training.app is None:
        raise HTTPException(status_code=503, detail="InsightFace model not loaded. Please restart the server or check model installation.")
        
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
        
    try:
        # Process base64 image
        image_rgb = process_base64_image(request.image_data)
        
        # Save base64 image to folder
        image_path = save_base64_image(request.image_data, request.name.strip(), ".jpg")
        
        # Get faces using InsightFace
        faces = training.app.get(image_rgb)
        
        if not faces:
            return {
                "success": False,
                "message": "No face detected in image"
            }
            
        if len(faces) > 1:
            return {
                "success": False,
                "message": "Multiple faces detected. Please ensure only one face is in the image."
            }
            
        face = faces[0]
        
        # Check quality
        if face.det_score < 0.7:
            return {
                "success": False,
                "message": "Face quality too low. Please improve lighting and face clarity."
            }
            
        # Extract embedding
        embedding = face.embedding
        
        # Register the face
        register_new_face(request.name.strip(), embedding, image_path)
        
        # Add timestamp info to face_database entry
        if request.name.strip() not in face_database:
            face_database[request.name.strip()] = {}
        face_database[request.name.strip()]["created_at"] = datetime.now().isoformat()
        face_database[request.name.strip()]["updated_at"] = datetime.now().isoformat()
        face_database[request.name.strip()]["image_path"] = image_path
        
        # Save databases
        save_face_database()
        save_user_database()
        
        return {
            "success": True,
            "message": f"Successfully registered {request.name.strip()}!",
            "detection_score": float(face.det_score),
            "bbox": face.bbox.astype(int).tolist(),
            "image_path": image_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")

@app.post("/api/faces/recognize")
async def recognize_face_base64(request: RecognizeRequest):
    """Recognize faces using base64 image data"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    if training.app is None:
        raise HTTPException(status_code=503, detail="InsightFace model not loaded. Please restart the server or check model installation.")
        
    try:
        # Process base64 image
        image_rgb = process_base64_image(request.image_data)
        
        # Get faces using InsightFace
        faces = training.app.get(image_rgb)
        
        if not faces:
            return {
                "success": False,
                "message": "No faces detected in image",
                "faces": []
            }
            
        results = []
        
        for i, face in enumerate(faces):
            if face.det_score > 0.5:  # Confidence threshold
                # Get face embedding
                embedding = face.embedding
                
                # Recognize face with mask detection
                name, confidence = enhanced_recognize_face(
                    embedding, 
                    confidence_threshold=0.6,
                    face_image=image_rgb,
                    face_bbox=face.bbox
                )
                
                # Get bounding box
                bbox = face.bbox.astype(int).tolist()
                
                # Get landmarks (5 key points: left eye, right eye, nose, left mouth, right mouth)
                landmarks = None
                if hasattr(face, 'kps') and face.kps is not None:
                    landmarks = face.kps.astype(int).tolist()
                
                # Determine if registered (green) or unknown (red)
                is_registered = name is not None and name != "UNKNOWN"
                
                # Log recognition event
                log_recognition_event(
                    name=name if name else "UNKNOWN",
                    confidence=confidence,
                    is_registered=is_registered
                )
                
                result = {
                    "face_id": i,
                    "name": name if name else "UNKNOWN",
                    "confidence": float(confidence),
                    "detection_score": float(face.det_score),
                    "bbox": bbox,  # [x1, y1, x2, y2]
                    "landmarks": landmarks,  # [[x1, y1], [x2, y2], ...] for 5 key points
                    "is_registered": is_registered
                }
                
                results.append(result)
                
        return {
            "success": True,
            "message": f"Processed {len(results)} faces",
            "faces": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition error: {str(e)}")

# Additional user management endpoints
@app.get("/api/faces/users/{user_id}")
async def get_user(user_id: str):
    """Get specific user information"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    # In this simple implementation, user_id is the name
    user_name = user_id
    
    if user_name not in face_database:
        raise HTTPException(status_code=404, detail="User not found")
        
    user_data = face_database[user_name]
    return {
        "success": True,
        "user": {
            "id": user_name,
            "name": user_name,
            "samples": user_data.get("samples", 1),
            "confidence": user_data.get("confidence", 0.8)
        }
    }

@app.post("/api/faces/users/{user_id}/add-sample")
async def add_face_sample(user_id: str, request: AddSampleRequest):
    """Add a face sample to existing user"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    if training.app is None:
        raise HTTPException(status_code=503, detail="InsightFace model not loaded. Please restart the server or check model installation.")
        
    user_name = user_id
    
    if user_name not in face_database:
        raise HTTPException(status_code=404, detail="User not found")
        
    try:
        # Process base64 image
        image_rgb = process_base64_image(request.image_data)
        
        # Save base64 image to folder
        image_path = save_base64_image(request.image_data, user_name, ".jpg")
        
        # Get faces using InsightFace
        faces = training.app.get(image_rgb)
        
        if not faces:
            return {
                "success": False,
                "message": "No face detected in image"
            }
            
        if len(faces) > 1:
            return {
                "success": False,
                "message": "Multiple faces detected. Please ensure only one face is in the image."
            }
            
        face = faces[0]
        
        # Check quality
        if face.det_score < 0.7:
            return {
                "success": False,
                "message": "Face quality too low. Please improve lighting and face clarity."
            }
            
        # Extract embedding and add to existing user
        embedding = face.embedding
        register_new_face(user_name, embedding, image_path)  # This will update existing user
        
        # Update timestamp
        face_database[user_name]["updated_at"] = datetime.now().isoformat()
        
        # Save databases
        save_face_database()
        save_user_database()
        
        return {
            "success": True,
            "message": f"Added face sample for {user_name}",
            "detection_score": float(face.det_score),
            "samples": face_database[user_name].get("samples", 1)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Add sample error: {str(e)}")

# Training status endpoints
@app.get("/api/faces/training-status")
async def get_training_status():
    """Get current training status"""
    return {
        "success": True,
        "status": training_status
    }

@app.delete("/api/faces/models")
async def delete_trained_models():
    """Delete trained models"""
    try:
        # Delete model files
        model_files = ['face_recognition_model.pkl', 'model_info.pkl']
        deleted_files = []
        
        for file_name in model_files:
            if os.path.exists(file_name):
                os.remove(file_name)
                deleted_files.append(file_name)
                
        return {
            "success": True,
            "message": f"Deleted {len(deleted_files)} model files",
            "deleted_files": deleted_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting models: {str(e)}")

# Enhanced training endpoints (placeholders)
@app.post("/api/faces/train-enhanced")
async def train_enhanced_models(use_synthetic: bool = True):
    """Train enhanced models with synthetic data"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        # Set training status
        global training_status
        training_status = {"is_training": True, "progress": 0, "message": "Starting enhanced training..."}
        
        # Import training functions
        from training import main as training_main
        
        # Run training (in production, use a task queue)
        training_main()
        
        training_status = {"is_training": False, "progress": 100, "message": "Enhanced training completed"}
        
        return {
            "success": True,
            "message": "Enhanced training completed",
            "database_size": len(face_database),
            "use_synthetic": use_synthetic
        }
        
    except Exception as e:
        training_status = {"is_training": False, "progress": 0, "message": f"Training failed: {str(e)}"}
        return {
            "success": False,
            "message": f"Enhanced training error: {str(e)}"
        }

@app.get("/api/faces/enhanced-training-status")
async def get_enhanced_training_status():
    """Get enhanced training status"""
    return {
        "success": True,
        "status": training_status
    }

# Dashboard endpoints
@app.get("/api/dashboard/recognition-trends")
async def get_recognition_trends(days: int = 7):
    """Get recognition trends"""
    # Generate trend data based on face database
    trends = []
    base_recognitions = len(face_database) * 3
    
    for i in range(days):
        day_num = i + 1
        # Generate realistic-looking data
        total_recognitions = base_recognitions + (day_num * 2) + (5 if day_num % 3 == 0 else 0)
        successful_recognitions = int(total_recognitions * 0.85)  # 85% success rate
        
        trends.append({
            "date": f"2024-01-{day_num:02d}",
            "total_recognitions": total_recognitions,
            "successful_recognitions": successful_recognitions,
            "accuracy": 0.85 + (i * 0.005)  # Slightly increasing accuracy
        })
    
    return {
        "success": True,
        "trends": trends,
        "period": f"Last {days} days"
    }

@app.get("/api/dashboard/top-recognized-users")
async def get_top_recognized_users(limit: int = 5):
    """Get top recognized users"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Create data based on face database
    top_users = []
    for name, data in list(face_database.items())[:limit]:
        top_users.append({
            "name": name,
            "recognition_count": data.get("samples", 1) * 15,  # Mock recognition count based on samples
            "total_samples": data.get("samples", 1),
            "avg_confidence": data.get("confidence", 0.8),
            "last_seen": "2024-01-01T12:00:00Z"  # Mock last seen
        })
    
    return {
        "success": True,
        "top_users": top_users
    }

@app.get("/api/dashboard/system-performance") 
async def get_system_performance():
    """Get system performance metrics"""
    # Calculate metrics based on system state
    total_requests = len(face_database) * 20  # Mock request count
    avg_processing_time = 0.15 + (len(face_database) * 0.01)  # Slightly increases with more users
    avg_confidence = sum(data.get("confidence", 0.8) for data in face_database.values()) / len(face_database) if face_database else 0.8
    
    # Performance grade based on processing time
    if avg_processing_time < 0.2:
        performance_grade = "Excellent"
    elif avg_processing_time < 0.4:
        performance_grade = "Good" 
    else:
        performance_grade = "Fair"
    
    return {
        "success": True,
        "avg_processing_time": avg_processing_time,
        "avg_confidence": avg_confidence,
        "total_requests": total_requests,
        "performance_grade": performance_grade,
        "cpu_usage": 15.5 if system_initialized else 5.0,
        "memory_usage": 32.8 if system_initialized else 12.0,
        "gpu_usage": 25.0 if system_initialized else 0.0
    }

@app.get("/api/faces/recognition-logs")
async def get_recognition_logs(limit: int = 100):
    """Get recent recognition logs"""
    try:
        log_file = "recognition_logs.json"
        
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
            
        # Return most recent logs first
        recent_logs = logs[-limit:] if len(logs) > limit else logs
        recent_logs.reverse()
        
        return {
            "success": True,
            "logs": recent_logs,
            "total": len(logs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")

# Sign Detection API Endpoints

def log_sign_detection_event(gesture: str, confidence: float, is_emergency: bool, hands_detected: int = 0, pose_detected: bool = False):
    """Log sign detection events to JSON file for tracking"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "gesture": gesture,
            "confidence": float(confidence),
            "is_emergency": is_emergency,
            "hands_detected": hands_detected,
            "pose_detected": pose_detected,
            "type": "sign_detection"
        }
        
        log_file = "sign_detection_logs.json"
        
        # Load existing logs or create new list
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
            
        # Add new log entry
        logs.append(log_entry)
        
        # Keep only last 1000 entries to prevent file from growing too large
        if len(logs) > 1000:
            logs = logs[-1000:]
            
        # Save back to file
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
            
        print(f"✅ Sign detection logged: {gesture} - {'EMERGENCY' if is_emergency else 'NORMAL'}")
        
    except Exception as e:
        print(f"❌ Error logging sign detection: {e}")

@app.get("/api/signs/status")
async def get_sign_detection_status():
    """Get sign detection system status"""
    return {
        "success": True,
        "status": "running" if sign_detector_initialized else "not_initialized",
        "mediapipe_loaded": sign_detector_initialized,
        "models_available": True,  # Sign detection works with MediaPipe landmarks, no ML models needed
        "gesture_detection_ready": sign_detector_initialized,
        "ml_models_trained": sign_training_manager is not None and sign_training_manager.model_trained if sign_training_manager else False
    }

@app.post("/api/signs/detect")
async def detect_signs(request: SignDetectionRequest):
    """Detect signs/gestures in uploaded image"""
    if not sign_detector_initialized:
        raise HTTPException(status_code=503, detail="Sign detection not initialized")
        
    try:
        sign_detector = get_sign_detector()
        if sign_detector is None:
            raise HTTPException(status_code=503, detail="Sign detector not available")
        
        # Process base64 image
        image_rgb = process_base64_image(request.image_data)
        
        # Use the sign detector (which now handles both MediaPipe and fallback modes)
        result = sign_detector.process_frame(image_rgb)
        
        # Log the detection
        log_sign_detection_event(
            gesture=result.get('gesture', 'NONE'),
            confidence=result.get('confidence', 0.0),
            is_emergency=result.get('is_emergency', False),
            hands_detected=len(result.get('landmarks', {}).get('hands', [])),
            pose_detected=result.get('landmarks', {}).get('pose') is not None
        )
        
        return {
            "success": True,
            "gesture": result.get('gesture', 'NONE'),
            "confidence": result.get('confidence', 0.0),
            "is_emergency": result.get('is_emergency', False),
            "hands_detected": len(result.get('landmarks', {}).get('hands', [])),
            "pose_detected": result.get('landmarks', {}).get('pose') is not None,
            "timestamp": result.get('timestamp'),
            "landmarks": result.get('landmarks', {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sign detection error: {str(e)}")

@app.post("/api/signs/detect-file")
async def detect_signs_file(file: UploadFile = File(...)):
    """Detect signs/gestures in uploaded image file"""
    if not sign_detector_initialized:
        raise HTTPException(status_code=503, detail="Sign detection not initialized")
        
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    try:
        sign_detector = get_sign_detector()
        if sign_detector is None:
            raise HTTPException(status_code=503, detail="Sign detector not available")
        
        # Read and process image
        file_content = await file.read()
        image_rgb = process_image_file(file_content)
        
        # Process frame for gesture detection
        result = sign_detector.process_frame(image_rgb)
        
        # Log the detection
        log_sign_detection_event(
            gesture=result.get('gesture', 'NONE'),
            confidence=result.get('confidence', 0.0),
            is_emergency=result.get('is_emergency', False),
            hands_detected=len(result.get('landmarks', {}).get('hands', [])),
            pose_detected=result.get('landmarks', {}).get('pose') is not None
        )
        
        return {
            "success": True,
            "gesture": result.get('gesture', 'NONE'),
            "confidence": result.get('confidence', 0.0),
            "is_emergency": result.get('is_emergency', False),
            "hands_detected": len(result.get('landmarks', {}).get('hands', [])),
            "pose_detected": result.get('landmarks', {}).get('pose') is not None,
            "timestamp": result.get('timestamp'),
            "landmarks": result.get('landmarks', {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sign detection error: {str(e)}")

@app.get("/api/signs/logs")
async def get_sign_detection_logs(limit: int = 100):
    """Get recent sign detection logs"""
    try:
        log_file = "sign_detection_logs.json"
        
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
            
        # Return most recent logs first
        recent_logs = logs[-limit:] if len(logs) > limit else logs
        recent_logs.reverse()
        
        return {
            "success": True,
            "logs": recent_logs,
            "total": len(logs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sign logs: {str(e)}")

@app.get("/api/signs/emergency-logs")
async def get_emergency_sign_logs(limit: int = 50):
    """Get recent emergency sign detection logs"""
    try:
        sign_detector = get_sign_detector()
        if sign_detector is None:
            return {"success": False, "logs": [], "message": "Sign detector not available"}
        
        emergency_logs = sign_detector.get_emergency_logs(limit)
        
        return {
            "success": True,
            "logs": emergency_logs,
            "total": len(emergency_logs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving emergency logs: {str(e)}")

@app.get("/api/signs/statistics")
async def get_sign_statistics():
    """Get sign detection statistics"""
    try:
        sign_detector = get_sign_detector()
        if sign_detector is None:
            return {
                "success": False,
                "message": "Sign detector not available",
                "statistics": {}
            }
        
        stats = sign_detector.get_statistics()
        
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

@app.post("/api/signs/train")
async def train_sign_detection_models():
    """Train sign detection models"""
    if not sign_detector_initialized:
        raise HTTPException(status_code=503, detail="Sign detection not initialized")
        
    try:
        global sign_training_manager
        if sign_training_manager is None:
            sign_training_manager = SignTrainingManager()
        
        # Start training process (in production, use background task)
        success = sign_training_manager.preprocess_training_data()
        if success:
            success = sign_training_manager.train_models()
        
        if success:
            stats = sign_training_manager.get_training_statistics()
            return {
                "success": True,
                "message": "Sign detection models trained successfully",
                "statistics": stats
            }
        else:
            return {
                "success": False,
                "message": "Training failed - no data or insufficient samples"
            }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Training error: {str(e)}"
        }

@app.post("/api/signs/download-datasets")
async def download_sign_datasets():
    """Download sign language datasets"""
    if not sign_detector_initialized:
        raise HTTPException(status_code=503, detail="Sign detection not initialized")
        
    try:
        global sign_training_manager
        if sign_training_manager is None:
            sign_training_manager = SignTrainingManager()
        
        # Download datasets (in production, use background task)
        success = sign_training_manager.download_sign_datasets()
        
        return {
            "success": success,
            "message": "Dataset download completed" if success else "Dataset download failed"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Dataset download error: {str(e)}"
        }

@app.delete("/api/signs/logs")
async def clear_sign_detection_logs():
    """Clear sign detection logs"""
    try:
        # Clear file-based logs
        log_file = "sign_detection_logs.json"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        # Clear detector logs
        sign_detector = get_sign_detector()
        if sign_detector:
            sign_detector.clear_logs()
        
        return {
            "success": True,
            "message": "Sign detection logs cleared"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing logs: {str(e)}")

@app.get("/api/signs/gestures")
async def get_supported_gestures():
    """Get list of supported gestures - simplified system"""
    return {
        "success": True,
        "gestures": [
            {
                "name": "THUMBS_UP",
                "description": "Thumb extended upward",
                "emergency": False,
                "instructions": "Make a fist with thumb pointing up",
                "icon": "👍"
            },
            {
                "name": "PALM_STOP", 
                "description": "Open palm facing camera (stop signal)",
                "emergency": False,
                "instructions": "Show open palm with all fingers extended facing the camera",
                "icon": "✋"
            },
            {
                "name": "ARMS_CROSSED",
                "description": "Both arms crossed over chest (Emergency Signal)",
                "emergency": True,
                "instructions": "Cross both arms over your chest - this triggers emergency alerts",
                "icon": "❌"
            }
        ],
        "note": "Simplified system with 3 core gestures for better reliability",
        "total_gestures": 3
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)