#!/usr/bin/env python3
"""
Sign Detection Training Module
Handles dataset management and ML model training for gesture recognition
"""

import os
import cv2
import numpy as np
import pickle
import glob
import sys
import subprocess
import zipfile
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

# ML and data processing imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import joblib

# MediaPipe and sign detection
import mediapipe as mp
from sign_detection import SimplifiedSignDetector, initialize_sign_detector

# Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignTrainingManager:
    """Manages sign detection dataset processing and model training"""
    
    def __init__(self):
        self.sign_detector = None
        self.training_data = []
        self.training_labels = []
        self.training_features = []
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        self.model_trained = False
        
        # Training parameters
        self.confidence_threshold = 0.7
        self.feature_size = 63  # 21 hand landmarks * 3 coordinates = 63 features per hand
        
        # Dataset paths
        self.dataset_dir = "sign_datasets"
        self.models_dir = "sign_models"
        
        # Create directories
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info("SignTrainingManager initialized")
    
    def initialize_detector(self):
        """Initialize sign detector for training"""
        try:
            if not initialize_sign_detector():
                raise Exception("Failed to initialize sign detector")
            
            from sign_detection import get_sign_detector
            self.sign_detector = get_sign_detector()
            logger.info("Sign detector initialized for training")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing sign detector: {e}")
            return False
    
    def download_sign_datasets(self):
        """Download sign language and gesture datasets using Kaggle API"""
        logger.info("Starting sign detection dataset download process...")
        
        datasets_to_download = [
            # {
            #     "dataset": "grassknoted/asl-alphabet",
            #     "filename": "asl-alphabet.zip",
            #     "extract_to": "asl_alphabet",
            #     "description": "ASL Alphabet Dataset"
            # },
            {
                "dataset": "ayuraj/asl-dataset",
                "filename": "asl-dataset.zip",
                "extract_to": "asl_gestures",
                "description": "ASL Gesture Dataset"
            },
            {
                "dataset": "datamunge/sign-language-mnist",
                "filename": "sign-language-mnist.zip",
                "extract_to": "sign_language_mnist",
                "description": "Sign Language MNIST"
            },
            {
                "dataset": "muhammadkhalid/sign-language-for-numbers",
                "filename": "sign-language-numbers.zip",
                "extract_to": "sign_numbers",
                "description": "Sign Language Numbers"
            }
        ]
        
        try:
            # Import and setup Kaggle API
            try:
                import kaggle
                logger.info("✅ Kaggle API available")
            except ImportError:
                logger.info("📦 Installing Kaggle API...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
                import kaggle
            
            # Setup Kaggle credentials from existing training.py
            logger.info("💡 Setting up demo credentials...")
            os.environ['KAGGLE_USERNAME'] = 'soundaryats24cse'
            os.environ['KAGGLE_KEY'] = 'ed8d7faa7b233596c35babba8761c00b'
            logger.info("✅ Demo Kaggle credentials configured")
            
            # Authenticate
            try:
                kaggle.api.authenticate()
                logger.info("✅ Kaggle authentication successful!")
            except Exception as e:
                logger.error(f"❌ Kaggle authentication failed: {e}")
                return self._create_sample_sign_structure()
            
            # Download datasets
            downloaded_count = 0
            for dataset_info in datasets_to_download:
                dataset_name = dataset_info["dataset"]
                filename = dataset_info["filename"]
                extract_to = dataset_info["extract_to"]
                description = dataset_info["description"]
                
                dataset_path = os.path.join(self.dataset_dir, extract_to)
                zip_path = os.path.join(self.dataset_dir, filename)
                
                # Skip if already exists
                if os.path.exists(dataset_path):
                    logger.info(f"⏭️  Dataset already exists: {extract_to}")
                    downloaded_count += 1
                    continue
                
                try:
                    logger.info(f"\n📥 Downloading {description}...")
                    logger.info(f"🔗 Dataset: {dataset_name}")
                    
                    # Download with progress indicator
                    download_complete = threading.Event()
                    
                    def show_progress():
                        chars = "|/-\\"
                        idx = 0
                        while not download_complete.is_set():
                            print(f"\r   └── Downloading... {chars[idx % len(chars)]}", end="", flush=True)
                            idx += 1
                            time.sleep(0.2)
                        print("\r   └── Download completed!     ")
                    
                    progress_thread = threading.Thread(target=show_progress)
                    progress_thread.start()
                    
                    # Download dataset
                    kaggle.api.dataset_download_files(
                        dataset_name,
                        path=self.dataset_dir,
                        unzip=False
                    )
                    
                    download_complete.set()
                    progress_thread.join()
                    
                    # Extract if download successful
                    if os.path.exists(zip_path):
                        logger.info(f"📂 Extracting {filename}...")
                        
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(dataset_path)
                        
                        # Clean up zip file
                        os.remove(zip_path)
                        
                        file_count = sum(len(files) for _, _, files in os.walk(dataset_path))
                        logger.info(f"✅ {description} extracted! ({file_count} files)")
                        downloaded_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to download {dataset_name}: {e}")
                    if download_complete:
                        download_complete.set()
            
            if downloaded_count > 0:
                logger.info(f"✅ Downloaded {downloaded_count} sign datasets!")
                return True
            else:
                logger.warning("⚠️ No datasets downloaded, creating sample structure...")
                return self._create_sample_sign_structure()
                
        except Exception as e:
            logger.error(f"⚠️ Dataset download error: {e}")
            return self._create_sample_sign_structure()
    
    def _create_sample_sign_structure(self):
        """Create sample sign dataset structure"""
        logger.info("📊 Creating sample sign dataset structure...")
        
        try:
            # Create sample directory structure
            sample_gestures = [
                'thumbs_up', 'thumbs_down', 'ok', 'peace', 'fist',
                'stop', 'help', 'call', 'point_left', 'point_right',
                'wave', 'cover_mouth', 'hands_up'
            ]
            
            dirs_created = []
            for gesture in sample_gestures:
                gesture_dir = os.path.join(self.dataset_dir, 'sample_gestures', gesture)
                os.makedirs(gesture_dir, exist_ok=True)
                dirs_created.append(gesture_dir)
            
            logger.info(f"✅ Created {len(dirs_created)} sample gesture directories")
            logger.info("📋 To add real training data:")
            logger.info("1. Place gesture images in the created directories")
            logger.info("2. Use .jpg, .png, .jpeg formats")
            logger.info("3. Ensure images contain clear hand gestures")
            logger.info("4. Re-run training after adding images")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Sample structure creation failed: {e}")
            return False
    
    def extract_hand_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract MediaPipe hand features from image"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract landmarks using sign detector
            if self.sign_detector is None:
                return None
                
            landmarks_data = self.sign_detector.extract_landmarks(image)
            hands = landmarks_data.get('hands', [])
            
            if not hands:
                return None
            
            # Use first hand for training
            hand_landmarks = hands[0]['landmarks']
            
            # Convert landmarks to feature vector
            features = []
            for landmark in hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None
    
    def process_dataset_directory(self, dataset_path: str, max_images_per_class: int = 20) -> Tuple[List, List]:
        """Process a dataset directory and extract features"""
        logger.info(f"📂 Processing dataset: {dataset_path}")
        
        features = []
        labels = []
        
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            return features, labels
        
        # Find all image files organized by class/gesture
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        class_counts = defaultdict(int)
        
        # Walk through directory structure
        for root, dirs, files in os.walk(dataset_path):
            # Skip if this is the root directory
            if root == dataset_path:
                continue
                
            # Get class name from directory
            class_name = os.path.basename(root).lower()
            
            # Find images in this class directory
            class_images = []
            for ext in image_extensions:
                class_images.extend(glob.glob(os.path.join(root, ext)))
                class_images.extend(glob.glob(os.path.join(root, ext.upper())))
            
            if not class_images:
                continue
            
            logger.info(f"📊 Processing class '{class_name}': {len(class_images)} images")
            
            # Limit images per class for super fast training (reduced to 20 for 5-10 min processing)
            if len(class_images) > max_images_per_class:
                # Take every nth image for better distribution
                step = len(class_images) // max_images_per_class
                class_images = class_images[::max(1, step)][:max_images_per_class]
            
            # Extract features from images
            processed_count = 0
            for img_path in class_images:
                feature_vector = self.extract_hand_features(img_path)
                
                if feature_vector is not None and len(feature_vector) == self.feature_size:
                    features.append(feature_vector)
                    labels.append(class_name)
                    processed_count += 1
                    
                    # Progress indicator (reduced frequency)
                    if processed_count % 25 == 0 or processed_count == len(class_images):
                        logger.info(f"   └── Processed {processed_count}/{len(class_images)} images")
            
            class_counts[class_name] = processed_count
            logger.info(f"✅ Class '{class_name}': {processed_count} features extracted")
        
        logger.info(f"📊 Dataset processing summary:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count} samples")
        
        return features, labels
    
    def preprocess_training_data(self):
        """Process all datasets and prepare training data"""
        logger.info("🔄 Preprocessing sign detection training data...")
        
        if not self.initialize_detector():
            logger.error("❌ Cannot initialize sign detector for training")
            return False
        
        all_features = []
        all_labels = []
        
        # Process each dataset directory
        dataset_subdirs = [
            'asl_alphabet',
            'asl_gestures', 
            'sign_language_mnist',
            'sign_numbers',
            'sample_gestures'
        ]
        
        for subdir in dataset_subdirs:
            dataset_path = os.path.join(self.dataset_dir, subdir)
            if os.path.exists(dataset_path):
                features, labels = self.process_dataset_directory(dataset_path)
                all_features.extend(features)
                all_labels.extend(labels)
        
        if not all_features:
            logger.warning("⚠️ No training features extracted from datasets")
            return False
        
        # Convert to numpy arrays
        self.training_features = np.array(all_features)
        self.training_labels = np.array(all_labels)
        
        # Encode labels
        self.training_labels = self.label_encoder.fit_transform(self.training_labels)
        
        # Scale features
        self.training_features = self.scaler.fit_transform(self.training_features)
        
        logger.info(f"📊 Training data prepared:")
        logger.info(f"  Features shape: {self.training_features.shape}")
        logger.info(f"  Unique classes: {len(self.label_encoder.classes_)}")
        logger.info(f"  Class distribution:")
        
        class_counts = Counter(self.training_labels)
        for class_idx, count in class_counts.items():
            class_name = self.label_encoder.classes_[class_idx]
            logger.info(f"    {class_name}: {count} samples")
        
        return True
    
    def train_models(self):
        """Train multiple ML models for gesture classification"""
        logger.info("🤖 Training sign detection models...")
        
        if len(self.training_features) == 0:
            logger.error("❌ No training data available")
            return False
        
        if len(set(self.training_labels)) < 2:
            logger.error("❌ Need at least 2 classes for training")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.training_features, 
            self.training_labels,
            test_size=0.2, 
            random_state=42,
            stratify=self.training_labels
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Train optimized models for faster training
        models_to_train = {
            'random_forest': RandomForestClassifier(
                n_estimators=50,  # Reduced from 100
                max_depth=10,     # Reduced from 15
                random_state=42,
                n_jobs=-1
            ),
            # Skip SVM as it's slower for large datasets
            # 'svm': SVC(
            #     kernel='rbf',
            #     probability=True,
            #     random_state=42
            # ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(64, 32),  # Reduced from (128, 64)
                max_iter=200,  # Reduced from 500
                random_state=42,
                early_stopping=True,  # Add early stopping
                validation_fraction=0.1
            )
        }
        
        best_model = None
        best_accuracy = 0
        best_model_name = ""
        
        for model_name, model in models_to_train.items():
            try:
                logger.info(f"🔄 Training {model_name}...")
                
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                logger.info(f"✅ {model_name}:")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Training time: {training_time:.2f}s")
                
                # Save model
                model_path = os.path.join(self.models_dir, f"{model_name}_sign_model.pkl")
                joblib.dump(model, model_path)
                
                self.models[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'path': model_path
                }
                
                # Track best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {e}")
        
        if best_model is None:
            logger.error("❌ No models trained successfully")
            return False
        
        # Save best model as primary
        best_model_path = os.path.join(self.models_dir, "best_sign_model.pkl")
        joblib.dump(best_model, best_model_path)
        
        # Save preprocessing objects
        scaler_path = os.path.join(self.models_dir, "sign_scaler.pkl")
        encoder_path = os.path.join(self.models_dir, "sign_label_encoder.pkl")
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save model metadata
        metadata = {
            'best_model': best_model_name,
            'best_accuracy': best_accuracy,
            'classes': self.label_encoder.classes_.tolist(),
            'feature_size': self.feature_size,
            'training_samples': len(self.training_features),
            'trained_at': datetime.now().isoformat(),
            'models': {
                name: {
                    'accuracy': info['accuracy'],
                    'training_time': info['training_time'],
                    'path': info['path']
                }
                for name, info in self.models.items()
            }
        }
        
        metadata_path = os.path.join(self.models_dir, "sign_models_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.model_trained = True
        
        logger.info(f"🏆 Best model: {best_model_name} (accuracy: {best_accuracy:.4f})")
        logger.info("✅ Sign detection models training completed!")
        
        return True
    
    def evaluate_models(self):
        """Evaluate trained models with detailed metrics"""
        if not self.models:
            logger.warning("No models to evaluate")
            return
        
        logger.info("📊 Evaluating trained models...")
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            self.training_features,
            self.training_labels,
            test_size=0.2,
            random_state=42,
            stratify=self.training_labels
        )
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            
            logger.info(f"\n📋 {model_name.upper()} Evaluation:")
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Accuracy: {accuracy:.4f}")
            
            # Classification report
            class_names = self.label_encoder.classes_
            report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
            
            logger.info("Per-class metrics:")
            for class_name in class_names:
                if class_name in report:
                    metrics = report[class_name]
                    logger.info(f"  {class_name}: precision={metrics['precision']:.3f}, "
                              f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")
    
    def load_trained_models(self):
        """Load previously trained models"""
        try:
            # Load metadata
            metadata_path = os.path.join(self.models_dir, "sign_models_metadata.json")
            if not os.path.exists(metadata_path):
                logger.info("No trained models found")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load preprocessing objects
            scaler_path = os.path.join(self.models_dir, "sign_scaler.pkl")
            encoder_path = os.path.join(self.models_dir, "sign_label_encoder.pkl")
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
            
            # Load best model
            best_model_path = os.path.join(self.models_dir, "best_sign_model.pkl")
            if os.path.exists(best_model_path):
                best_model = joblib.load(best_model_path)
                self.models['best'] = {
                    'model': best_model,
                    'accuracy': metadata.get('best_accuracy', 0.0),
                    'path': best_model_path
                }
                self.model_trained = True
            
            logger.info(f"✅ Loaded sign detection models:")
            logger.info(f"  Best model: {metadata.get('best_model', 'Unknown')}")
            logger.info(f"  Accuracy: {metadata.get('best_accuracy', 0.0):.4f}")
            logger.info(f"  Classes: {len(metadata.get('classes', []))}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading trained models: {e}")
            return False
    
    def predict_gesture(self, image_path: str) -> Tuple[str, float]:
        """Predict gesture from image using trained model"""
        if not self.model_trained or 'best' not in self.models:
            return "NO_MODEL", 0.0
        
        try:
            # Extract features
            features = self.extract_hand_features(image_path)
            if features is None:
                return "NO_HAND_DETECTED", 0.0
            
            # Preprocess
            features = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Predict
            model = self.models['best']['model']
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Get class name and confidence
            class_name = self.label_encoder.classes_[prediction]
            confidence = probabilities[prediction]
            
            return class_name.upper(), confidence
            
        except Exception as e:
            logger.error(f"Error predicting gesture: {e}")
            return "PREDICTION_ERROR", 0.0
    
    def get_training_statistics(self) -> Dict:
        """Get training statistics"""
        if not hasattr(self, 'training_features') or len(self.training_features) == 0:
            return {
                'total_samples': 0,
                'total_classes': 0,
                'models_trained': 0,
                'best_accuracy': 0.0
            }
        
        metadata_path = os.path.join(self.models_dir, "sign_models_metadata.json")
        best_accuracy = 0.0
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                best_accuracy = metadata.get('best_accuracy', 0.0)
            except:
                pass
        
        return {
            'total_samples': len(self.training_features),
            'total_classes': len(self.label_encoder.classes_) if hasattr(self, 'label_encoder') else 0,
            'models_trained': len(self.models),
            'best_accuracy': best_accuracy,
            'feature_size': self.feature_size
        }

def main():
    """Main sign detection training function"""
    print("🤟 Sign Detection Training System")
    print("=" * 60)
    print("📋 This will:")
    print("   1. Download sign language datasets from Kaggle")
    print("   2. Extract MediaPipe hand landmarks from images")
    print("   3. Train ML models for gesture classification")
    print("   4. Save trained models for production use")
    print("=" * 60)
    
    # Initialize trainer
    trainer = SignTrainingManager()
    
    try:
        # Step 1: Download datasets
        print("\n📋 Step 1: Downloading Sign Language Datasets")
        if not trainer.download_sign_datasets():
            print("⚠️ Dataset download failed, but continuing with available data...")
        
        # Step 2: Preprocess data
        print("\n📋 Step 2: Preprocessing Training Data")
        if not trainer.preprocess_training_data():
            print("❌ No training data available")
            return
        
        # Step 3: Train models
        print("\n📋 Step 3: Training ML Models")
        if trainer.train_models():
            print("✅ Model training completed successfully!")
            
            # Step 4: Evaluate models
            print("\n📋 Step 4: Evaluating Models")
            trainer.evaluate_models()
            
            # Show statistics
            stats = trainer.get_training_statistics()
            print(f"\n🎉 SIGN DETECTION TRAINING COMPLETED!")
            print("=" * 60)
            print(f"📊 Training Statistics:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Total classes: {stats['total_classes']}")
            print(f"  Models trained: {stats['models_trained']}")
            print(f"  Best accuracy: {stats['best_accuracy']:.4f}")
            print(f"\n🚀 Models saved to: {trainer.models_dir}/")
            print(f"💡 Ready for integration with sign detection API!")
            
        else:
            print("❌ Model training failed")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()