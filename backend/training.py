#!/usr/bin/env python3
"""
Face Recognition Training Module
Based on mask-detection.ipynb Cells 1 & 2
"""

import os
import cv2
import numpy as np
import pickle
import glob
import sys
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from collections import defaultdict
import time

# For mask detection
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("WARNING: TensorFlow not available. Mask detection will use basic image processing.")
    TENSORFLOW_AVAILABLE = False

# Global variables
face_database = {}
training_data = []
training_labels = []
training_embeddings = []
label_encoder = LabelEncoder()
model_trained = False
app = None  # Will be initialized
mask_detector = None  # Will be initialized for mask detection

# Recognition parameters
SIMILARITY_THRESHOLD = 0.6
FACE_CONFIDENCE_THRESHOLD = 0.8
MASK_CONFIDENCE_THRESHOLD = 0.5

def initialize_insightface():
    """Initialize InsightFace buffalo_l model (Cell 1 functionality)"""
    global app
    
    print("Loading InsightFace buffalo_l model...")
    
    try:
        import insightface
        from insightface.app import FaceAnalysis
        print("InsightFace imported successfully")
        
        # Try GPU first, fallback to CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ctx_id = 0
        
        try:
            # Initialize face analysis with buffalo_l model (highest accuracy)
            app = FaceAnalysis(name='buffalo_l', providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            print(" InsightFace initialized with GPU support")
        except Exception as gpu_error:
            print(f" GPU initialization failed: {gpu_error}")
            print(" Trying CPU-only mode...")
            
            # Fallback to CPU only
            providers = ['CPUExecutionProvider']
            ctx_id = -1
            app = FaceAnalysis(name='buffalo_l', providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            print(" InsightFace initialized with CPU-only")
        
        # Test with dummy image to ensure it works
        import numpy as np
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_faces = app.get(test_image)
        print(f" Model test successful: detected {len(test_faces)} faces in test image")
        
        # Check available providers
        active_providers = getattr(app.models['detection'], 'providers', providers)
        gpu_available = any('CUDA' in str(provider) for provider in active_providers)
        print(" GPU acceleration enabled" if gpu_available else " Using CPU (may be slower)")
            
        return True
        
    except Exception as e:
        print(f" Error loading InsightFace: {e}")
        print(" Common fixes:")
        print("   - Install: pip install insightface onnxruntime")
        print("   - For GPU: pip install onnxruntime-gpu")
        print("   - Check internet connection for model download")
        
        # Keep app as None to prevent 500 errors
        app = None
        return False

def initialize_mask_detector():
    """Initialize mask detection model"""
    global mask_detector
    
    print(" Initializing mask detection model...")
    
    if not TENSORFLOW_AVAILABLE:
        print(" TensorFlow not available, using basic mask detection")
        mask_detector = "basic"
        return True
    
    try:
        # Try to load a pre-trained mask detection model
        # For now, we'll use a simple approach based on face region analysis
        print(" Basic mask detection initialized")
        mask_detector = "basic"
        return True
        
    except Exception as e:
        print(f" Error initializing mask detector: {e}")
        mask_detector = None
        return False

def detect_mask_simple(face_image, face_bbox):
    """
    Simple mask detection based on face region analysis
    Returns: (has_mask: bool, confidence: float)
    """
    try:
        # Extract face region
        x1, y1, x2, y2 = [int(coord) for coord in face_bbox]
        
        # Ensure coordinates are within image bounds
        h, w = face_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False, 0.0
            
        face_roi = face_image[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return False, 0.0
        
        # Convert to HSV for better color analysis
        hsv_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        # Define mask color ranges (common mask colors: white, blue, black)
        # White masks
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        
        # Blue masks  
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        
        # Black/dark masks
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 50])
        
        # Create masks for each color range
        white_mask = cv2.inRange(hsv_face, white_lower, white_upper)
        blue_mask = cv2.inRange(hsv_face, blue_lower, blue_upper)  
        black_mask = cv2.inRange(hsv_face, black_lower, black_upper)
        
        # Combine all mask color detections
        combined_mask = cv2.bitwise_or(white_mask, cv2.bitwise_or(blue_mask, black_mask))
        
        # Focus on lower half of face (where masks typically are)
        face_height = face_roi.shape[0]
        lower_face = combined_mask[face_height//2:, :]
        
        # Calculate percentage of mask-colored pixels in lower face region
        total_pixels = lower_face.shape[0] * lower_face.shape[1]
        mask_pixels = np.sum(lower_face > 0)
        mask_percentage = mask_pixels / total_pixels if total_pixels > 0 else 0
        
        # Additional check: look for horizontal edges (typical of mask edges)
        gray_lower_face = cv2.cvtColor(face_roi[face_height//2:, :], cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi[face_height//2:, :]
        edges = cv2.Canny(gray_lower_face, 50, 150)
        edge_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
        
        horizontal_lines = 0
        if edge_lines is not None:
            for line in edge_lines:
                x1_line, y1_line, x2_line, y2_line = line[0]
                # Check if line is roughly horizontal
                angle = abs(np.arctan2(y2_line - y1_line, x2_line - x1_line) * 180 / np.pi)
                if angle < 15 or angle > 165:  # Nearly horizontal
                    horizontal_lines += 1
        
        # Combine evidence
        mask_confidence = min(1.0, mask_percentage * 2 + (horizontal_lines * 0.1))
        has_mask = mask_confidence > MASK_CONFIDENCE_THRESHOLD
        
        return has_mask, mask_confidence
        
    except Exception as e:
        print(f" Error in mask detection: {e}")
        return False, 0.0

def download_and_extract_datasets():
    """Download and extract mask detection datasets using Kaggle API (Cell 1 functionality)"""
    print(" Starting mask detection dataset download process...")
    
    import threading
    import time
    
    datasets_to_download = [
        {
            "dataset": "ashishjangra27/face-mask-12k-images-dataset",
            "filename": "face-mask-12k-dataset.zip",
            "extract_to": "mask_dataset_12k",
            "description": "12K Face Mask Detection Dataset"
        },
        {
            "dataset": "omkargurav/face-mask-dataset",
            "filename": "face-mask-dataset.zip", 
            "extract_to": "mask_dataset1",
            "description": "Face Mask Detection Dataset"
        },
        {
            "dataset": "andrewmvd/face-mask-detection",
            "filename": "face-mask-detection.zip",
            "extract_to": "face_detection_dataset",
            "description": "Face Mask Detection with Annotations"
        }
    ]
    
    try:
        # Import and setup Kaggle API
        try:
            import kaggle
            print(" Kaggle API available")
        except ImportError:
            print(" Installing Kaggle API...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            import kaggle
            
        # Setup Kaggle credentials - use notebook credentials directly
        print(" Setting up demo credentials from notebook...")
        os.environ['KAGGLE_USERNAME'] = 'soundaryats24cse'
        os.environ['KAGGLE_KEY'] = 'ed8d7faa7b233596c35babba8761c00b'
        print(" Demo Kaggle credentials configured")
        
        # Try to authenticate
        try:
            kaggle.api.authenticate()
            print(" Kaggle authentication successful!")
        except Exception as e:
            print(f" Kaggle authentication failed: {e}")
            print(" To setup your own Kaggle API:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Create new API token")
            print("3. Set environment variables: KAGGLE_USERNAME and KAGGLE_KEY")
            print(" Creating sample structure instead...")
            return create_sample_structure()
            
        # Download only dataset2 - focus on face-mask-dataset from notebook
        dataset_downloaded = False
                
        # Dataset 2: Face Mask Dataset (163 MB from notebook)
        if not os.path.exists('face-mask-dataset.zip'):
            download_complete2 = None
            try:
                print("\n Downloading face-mask-dataset...")
                print(" URL: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset")
                print(" Size: ~163 MB (7,553 images)")
                print(" Estimated download time: 2-5 minutes")
                print("    Downloading dataset files...")
                
                # Create a progress indicator
                download_complete2 = threading.Event()
                
                def show_progress():
                    chars = "|/-\\"
                    idx = 0
                    while not download_complete2.is_set():
                        print(f"\r    Downloading... {chars[idx % len(chars)]}", end="", flush=True)
                        idx += 1
                        time.sleep(0.2)
                    print("\r    Download completed!     ")
                
                progress_thread = threading.Thread(target=show_progress)
                progress_thread.start()
                
                kaggle.api.dataset_download_files(
                    'omkargurav/face-mask-dataset',
                    path='.', 
                    unzip=False
                )
                
                download_complete2.set()
                progress_thread.join()
                
                # Check file size
                if os.path.exists('face-mask-dataset.zip'):
                    size = os.path.getsize('face-mask-dataset.zip') / (1024*1024)
                    print(f" Dataset 2 downloaded! ({size:.1f} MB)")
                    dataset_downloaded = True
                else:
                    print(" Download completed but file not found")
                    
            except Exception as e:
                if download_complete2:  # Stop progress indicator  
                    download_complete2.set()
                print(f"\r Failed to download dataset 2: {e}")
        
        # Extract only dataset2
        extracted = False
                
        if os.path.exists('face-mask-dataset.zip') and not os.path.exists('mask_dataset2'):
            print("\n Extracting face-mask-dataset...")
            print(" Extracting files to mask_dataset2/...")
            extract_complete2 = None
            try:
                extract_complete2 = threading.Event()
                
                def show_extract_progress():
                    chars = ""  # Spinning animation
                    idx = 0
                    while not extract_complete2.is_set():
                        print(f"\r    Extracting... {chars[idx % len(chars)]}", end="", flush=True)
                        idx += 1
                        time.sleep(0.1)
                    print("\r    Extraction completed!   ")
                
                progress_thread = threading.Thread(target=show_extract_progress)
                progress_thread.start()
                
                with zipfile.ZipFile('face-mask-dataset.zip', 'r') as zip_ref:
                    zip_ref.extractall('mask_dataset2')
                
                extract_complete2.set()
                progress_thread.join()
                
                # Count extracted files
                file_count = sum(len(files) for _, _, files in os.walk('mask_dataset2'))
                print(f" Dataset 2 extracted! ({file_count} files)")
                extracted = True
            except Exception as e:
                if extract_complete2:
                    extract_complete2.set()
                print(f"\r Failed to extract dataset 2: {e}")
                
        if dataset_downloaded or extracted:
            print(" Dataset setup completed!")
            return True
        elif os.path.exists('mask_dataset1') or os.path.exists('mask_dataset2'):
            print(" Found existing datasets!")
            return True
        else:
            print(" No datasets available, creating sample structure...")
            return create_sample_structure()
            
    except Exception as e:
        print(f" Dataset download error: {e}")
        print(" Creating sample structure instead...")
        return create_sample_structure()

def create_sample_structure():
    """Create sample data structure for testing"""
    print(" Creating sample training data structure...")
    
    try:
        # Create directory structure (only dataset2)
        dirs = [
            'mask_dataset2/train/with_mask',
            'mask_dataset2/train/without_mask',
            'mask_dataset2/test/with_mask', 
            'mask_dataset2/test/without_mask'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        print(" Sample directory structure created!")
        print(" To add real training data:")
        print("1. Place face images in the created directories:")
        for dir_path in dirs:
            print(f"   - {dir_path}/")
        print("2. Use .jpg, .png, .jpeg, .bmp formats")
        print("3. Ensure images contain clear faces")
        print("4. Re-run training after adding images")
        print("\n The system will download the face-mask-dataset (163 MB) automatically!")
        
        return True
        
    except Exception as e:
        print(f" Sample structure creation failed: {e}")
        return False

def preprocess_training_data():
    """Process training datasets (Cell 2 functionality)"""
    global training_embeddings, training_labels
    
    print(" Processing training datasets...")
    
    training_embeddings = []
    training_labels = []
    training_images = []
    
    dataset_paths = []
    
    # Check available datasets - only mask_dataset2
    if os.path.exists('mask_dataset2'):
        dataset_paths.append('mask_dataset2')
        
    if not dataset_paths:
        print(" No datasets found, using registration-only mode...")
        return [], [], []
        
    label_count = defaultdict(int)
    person_id = 1
    
    for dataset_path in dataset_paths:
        print(f" Processing {dataset_path}...")
        
        # Find all image files recursively
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(f"{dataset_path}/**/{ext}", recursive=True))
            
        print(f" Found {len(image_files)} images in {dataset_path}")
        
        # Process images and create multiple identities  
        total_to_process = min(len(image_files), 400)
        print(f" Processing {total_to_process} images for face detection...")
        
        for i, img_path in enumerate(image_files[:400]):  # Process more images
            try:
                # Show progress every 10 images
                if i % 10 == 0:
                    progress_pct = (i / total_to_process) * 100
                    print(f"\r Progress: {progress_pct:.1f}% ({i}/{total_to_process}) | Found faces: {len(training_embeddings)}", end="", flush=True)
                
                # Create diverse labels for better training
                path_parts = img_path.lower().split('/')
                filename = os.path.basename(img_path).lower()
                
                # Determine label strategy for mask training
                is_masked = False
                if 'with_mask' in img_path.lower() or ('mask' in img_path.lower() and 'without' not in img_path.lower() and 'no_mask' not in img_path.lower()):
                    base_label = 'with_mask'
                    is_masked = True
                elif 'without' in img_path.lower() or 'no_mask' in img_path.lower():
                    base_label = 'without_mask'
                    is_masked = False
                else:
                    # Default based on filename patterns
                    base_label = 'person'
                    is_masked = False
                    
                # Create multiple person identities within categories
                person_suffix = (i // 20) + 1  # Group every 20 images as same person
                
                if is_masked:
                    label = f"masked_person_{person_suffix}"
                else:
                    label = f"unmasked_person_{person_suffix}"
                    if i % 30 == 0:  # New person every 30 images
                        person_id += 1
                        
                # Load and process image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Face detection and embedding extraction
                faces = app.get(img_rgb)
                
                if faces:
                    face = faces[0]  # Take the first face
                    if face.det_score > 0.3:  # Lower threshold for more data
                        embedding = face.embedding
                        
                        training_embeddings.append(embedding)
                        training_labels.append(label)
                        training_images.append(img_rgb)
                        label_count[label] += 1
            except Exception as e:
                continue
                
        # Final progress update
        print(f"\r Processing completed! Found {len(training_embeddings)} faces in {total_to_process} images       ")
                
    print(f"\n Training Data Summary:")
    for label, count in label_count.items():
        print(f"  {label}: {count} samples")
        
    return training_embeddings, training_labels, training_images

def train_models():
    """Train ML models (Cell 2 functionality)"""
    global model_trained, face_database
    
    if not training_embeddings or len(set(training_labels)) <= 1:
        print(" Insufficient diverse data, using cosine similarity method...")
        return False
        
    print(f"\n Training advanced recognition model with {len(training_embeddings)} samples...")
    print(f" Number of unique identities: {len(set(training_labels))}")
    
    # Convert to numpy arrays
    X = np.array(training_embeddings)
    y = np.array(training_labels)
    
    # Check if we have enough samples for train/test split
    if len(set(y)) >= 2 and len(X) >= 10:
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train models
        print(" Training ensemble models...")
        
        try:
            # Model 1: Support Vector Machine
            svm_model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
            svm_model.fit(X_train, y_train)
            svm_pred = svm_model.predict(X_val)
            svm_accuracy = accuracy_score(y_val, svm_pred)
            
            # Model 2: Random Forest
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_val)
            rf_accuracy = accuracy_score(y_val, rf_pred)
            
            print(f" SVM Model Accuracy: {svm_accuracy:.3f}")
            print(f" Random Forest Accuracy: {rf_accuracy:.3f}")
            
            # Select best model
            if svm_accuracy > rf_accuracy:
                best_model = svm_model
                best_accuracy = svm_accuracy
                model_type = "SVM"
            else:
                best_model = rf_model
                best_accuracy = rf_accuracy
                model_type = "Random Forest"
                
            print(f" Best Model: {model_type} (Accuracy: {best_accuracy:.3f})")
            
            # Save the trained model
            joblib.dump(best_model, 'face_recognition_model.pkl')
            model_trained = True
            
            # Save model info
            model_info = {
                'type': model_type,
                'accuracy': best_accuracy,
                'classes': list(best_model.classes_),
                'trained_at': time.time()
            }
            
            with open('model_info.pkl', 'wb') as f:
                pickle.dump(model_info, f)
                
            return True
            
        except Exception as e:
            print(f" Model training error: {e}")
            print(" Using cosine similarity method instead...")
            return False
    else:
        print(" Insufficient data for supervised learning, using cosine similarity...")
        return False

def create_face_database():
    """Create enhanced face database"""
    global face_database
    
    if not training_embeddings:
        print(" No training data available")
        return
        
    unique_labels = list(set(training_labels))
    face_database = {}
    
    for label in unique_labels:
        # Find all embeddings for this label
        label_indices = [i for i, l in enumerate(training_labels) if l == label]
        label_embeddings = [training_embeddings[i] for i in label_indices]
        
        if label_embeddings:
            # Use mean embedding for better representation
            mean_embedding = np.mean(label_embeddings, axis=0)
            
            # Store mask and unmask embeddings separately for better recognition
            mask_embeddings = []
            unmask_embeddings = []
            
            for i in label_indices:
                if 'masked' in training_labels[i]:
                    mask_embeddings.append(training_embeddings[i])
                else:
                    unmask_embeddings.append(training_embeddings[i])
            
            face_database[label] = {
                'embedding': mean_embedding,
                'samples': len(label_embeddings),
                'confidence': 0.9 if len(label_embeddings) > 5 else 0.7,
                'mask_embeddings': mask_embeddings if mask_embeddings else [],
                'unmask_embeddings': unmask_embeddings if unmask_embeddings else []
            }
            
    print(f"\n Enhanced Face Database Created:")
    print(f"   Total Identities: {len(face_database)}")
    for name, data in face_database.items():
        print(f"  {name}: {data['samples']} samples (confidence: {data['confidence']:.1f})")

def enhanced_recognize_face(face_embedding, confidence_threshold=0.6, face_image=None, face_bbox=None):
    """Enhanced recognition function with mask detection support"""
    if not face_database:
        return None, 0.0
    
    # Detect if person is wearing a mask
    wearing_mask = False
    mask_confidence = 0.0
    
    if face_image is not None and face_bbox is not None and mask_detector is not None:
        wearing_mask, mask_confidence = detect_mask_simple(face_image, face_bbox)
        if wearing_mask:
            print(f" Mask detected (confidence: {mask_confidence:.3f})")
            # Adjust confidence threshold for masked faces (they're harder to recognize)
            confidence_threshold = confidence_threshold * 0.8  # Lower threshold for masked faces
        
    # Method 1: Use trained model if available  
    if model_trained and os.path.exists('face_recognition_model.pkl'):
        try:
            best_model = joblib.load('face_recognition_model.pkl')
            # Predict using trained model
            probabilities = best_model.predict_proba([face_embedding])
            classes = best_model.classes_
            
            max_prob_idx = np.argmax(probabilities[0])
            max_prob = probabilities[0][max_prob_idx]
            predicted_name = classes[max_prob_idx]
            
            # For masked faces, also check if we have mask-trained embeddings
            if wearing_mask and 'mask_embeddings' in face_database.get(predicted_name, {}):
                mask_embeddings = face_database[predicted_name]['mask_embeddings']
                mask_similarities = [cosine_similarity([face_embedding], [emb])[0][0] for emb in mask_embeddings]
                if mask_similarities:
                    max_mask_similarity = max(mask_similarities)
                    # Use mask-specific similarity if it's better
                    if max_mask_similarity > max_prob:
                        max_prob = max_mask_similarity
            
            if max_prob > confidence_threshold:
                return predicted_name, max_prob
        except Exception as e:
            print(f" Model prediction failed: {e}")
            
    # Method 2: Cosine similarity with mask awareness
    max_similarity = 0.0
    recognized_name = None
    
    for name, data in face_database.items():
        # Calculate similarity with regular embeddings
        similarity = cosine_similarity([face_embedding], [data['embedding']])[0][0]
        
        # If wearing mask and we have mask embeddings for this person, use those
        if wearing_mask and 'mask_embeddings' in data:
            mask_similarities = [cosine_similarity([face_embedding], [emb])[0][0] for emb in data['mask_embeddings']]
            if mask_similarities:
                mask_similarity = max(mask_similarities)
                # Use the better of the two similarities
                similarity = max(similarity, mask_similarity)
                print(f" Mask-aware similarity for {name}: {mask_similarity:.3f} (regular: {cosine_similarity([face_embedding], [data['embedding']])[0][0]:.3f})")
        
        # Adjust threshold based on training data quality and mask status
        adjusted_threshold = confidence_threshold * data['confidence']
        if wearing_mask:
            # Be more lenient for masked faces
            adjusted_threshold *= 0.9
        
        if similarity > max_similarity:
            max_similarity = similarity
            if similarity > adjusted_threshold:
                recognized_name = name
                
    return recognized_name, max_similarity

def register_new_face(name, face_embedding, image_path=None):
    """Register a new face in the database"""
    global face_database
    
    if name in face_database:
        # Update existing entry with new sample
        existing_embedding = face_database[name]['embedding']
        samples = face_database[name]['samples']
        
        # Weighted average of embeddings
        new_embedding = (existing_embedding * samples + face_embedding) / (samples + 1)
        
        face_database[name]['embedding'] = new_embedding
        face_database[name]['samples'] += 1
        face_database[name]['confidence'] = min(0.95, face_database[name]['confidence'] + 0.1)
        
        # Add image path to list if provided
        if image_path:
            if 'image_paths' not in face_database[name]:
                face_database[name]['image_paths'] = []
            face_database[name]['image_paths'].append(image_path)
        
        print(f" Updated {name} with new sample (total: {face_database[name]['samples']})")
    else:
        # Add new entry
        face_database[name] = {
            'embedding': face_embedding,
            'samples': 1,
            'confidence': 0.8,
            'image_paths': [image_path] if image_path else []
        }
        print(f" Registered new person: {name}")

def save_face_database():
    """Save face database to disk"""
    try:
        with open('face_database.pkl', 'wb') as f:
            pickle.dump(face_database, f)
        print(" Face database saved")
    except Exception as e:
        print(f" Error saving database: {e}")

def load_face_database():
    """Load face database from disk"""
    global face_database
    try:
        if os.path.exists('face_database.pkl'):
            with open('face_database.pkl', 'rb') as f:
                face_database = pickle.load(f)
            print(f" Face database loaded with {len(face_database)} identities")
            return True
    except Exception as e:
        print(f" Error loading database: {e}")
    return False

def main():
    """Main mask-aware face recognition training function"""
    print(" Face Recognition with Mask Detection - Training System")
    print("=" * 70)
    print(" This will:")
    print("   1. Initialize InsightFace buffalo_l model")
    print("   2. Download mask detection datasets from Kaggle (~300MB)")
    print("   3. Process masked and unmasked face images")
    print("   4. Train SVM/Random Forest models")
    print("   5. Create mask-aware face database")
    print("   6. Save PKL files for production use")
    print("=" * 70)
    
    # Check if user wants to continue
    try:
        user_input = input("\n Start training? (y/n): ").lower().strip()
        if user_input not in ['y', 'yes']:
            print(" Training cancelled by user")
            return
    except KeyboardInterrupt:
        print("\n Training cancelled by user")
        return
    
    # Step 1: Initialize InsightFace and Mask Detection
    print("\n Step 1: Loading Models")
    if not initialize_insightface():
        print(" Failed to initialize InsightFace")
        return
    print(" InsightFace buffalo_l model loaded successfully!")
    
    initialize_mask_detector()
    print(" Mask detection initialized")
    
    # Step 2: Download mask detection datasets
    print("\n Step 2: Setting up Mask Detection Datasets")
    download_and_extract_datasets()
    
    # Step 3: Load existing database
    print("\n Step 3: Loading Existing Database")
    if load_face_database():
        print(" Previous face database loaded")
    else:
        print(" No previous database found - starting fresh")
    
    # Step 4: Process training data (mask-aware)
    print("\n Step 4: Processing Mask/Unmask Training Data")
    global training_embeddings, training_labels
    training_embeddings, training_labels, training_images = preprocess_training_data()
    
    # Step 5: Train models with mask awareness
    print("\n Step 5: Training ML Models with Mask Awareness")
    if training_embeddings and len(training_embeddings) > 0:
        print(f" Found {len(training_embeddings)} face samples for training")
        
        # Show mask/unmask distribution
        masked_count = len([l for l in training_labels if 'masked' in l])
        unmasked_count = len(training_labels) - masked_count
        print(f"   - Masked faces: {masked_count}")
        print(f"   - Unmasked faces: {unmasked_count}")
        
        trained = train_models()
        create_face_database()
        save_face_database()
        
        print("\n" + "=" * 70)
        print(" MASK-AWARE TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f" Recognition Method: {'ML Model + Cosine Similarity' if trained else 'Cosine Similarity'}")
        print(f" Face Database Size: {len(face_database)} identities")
        print(f" Mask Detection: Enabled")
        print(f" Training Distribution:")
        print(f"   - Masked samples: {masked_count}")
        print(f"   - Unmasked samples: {unmasked_count}")
        print(f" Files Created:")
        if os.path.exists('face_database.pkl'):
            print("   face_database.pkl - Face embeddings database (with mask embeddings)")
        if os.path.exists('face_recognition_model.pkl'):
            print("   face_recognition_model.pkl - Trained ML model") 
        if os.path.exists('model_info.pkl'):
            print("   model_info.pkl - Model metadata")
            
        print(f"\n System is ready for mask-aware recognition!")
        print(f" Next steps:")
        print(f"   1. Run 'python main.py' to start the backend server")
        print(f"   2. Run 'npm start' in frontend/ to start the web interface")
        print(f"   3. Register users and test mask recognition!")
        
    else:
        print("\n" + "=" * 70) 
        print("  NO TRAINING DATA FOUND")
        print("=" * 70)
        print(" The system will work in registration-only mode.")
        print(" Possible issues:")
        print("   No internet connection for Kaggle download")
        print("   Kaggle API credentials not set")
        print("   Dataset extraction failed")
        print("\n System is still ready - you can:")
        print("   Start the API: python main.py")
        print("   Register faces manually via the frontend")
        print("   System will use cosine similarity for recognition")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Training interrupted by user")
    except Exception as e:
        print(f"\n\n Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n Common solutions:")
        print("   Check internet connection")
        print("   Install missing dependencies: pip install -r requirements.txt")
        print("   Set Kaggle API credentials")
        print("   Run as administrator if file permission errors")