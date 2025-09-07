# 🎭 Face Recognition with Mask Detection

A production-ready face recognition system that can identify people both with and without masks. Built with FastAPI backend and React frontend, powered by InsightFace buffalo_l model.

## ✨ Features

- **🎭 Mask-Aware Recognition**: Recognizes faces with AND without masks
- **🤖 Dual ML Models**: SVM + Random Forest with cosine similarity fallback
- **📱 Modern Web Interface**: React frontend with real-time recognition
- **📊 Comprehensive Dashboard**: Statistics, trends, and user management
- **🔄 Live Recognition**: Real-time webcam face recognition
- **💾 Persistent Storage**: Face embeddings and user data saved to PKL files
- **🎯 High Accuracy**: InsightFace buffalo_l model with 512-dim embeddings

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React.js      │    │    FastAPI       │    │   PKL Files     │
│   Frontend      │◄──►│    Backend       │◄──►│   Database      │
│   Port 3000     │    │    Port 8000     │    │   + JSON Logs   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        │
        │                        │                        │
   ┌─────────┐            ┌─────────────┐         ┌──────────────┐
   │ Webcam  │            │ InsightFace │         │ Kaggle       │
   │ Capture │            │ buffalo_l   │         │ Datasets     │
   └─────────┘            └─────────────┘         └──────────────┘
```

## 🚀 Quick Start for Developers

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git
- Internet connection (for Kaggle datasets)

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd face-recogn/model-1
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\\Scripts\\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run mask detection training (downloads ~300MB datasets)
python training.py

# Start backend server
python main.py
```

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### 4. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📋 Developer Workflow

### Standard Development Flow
```bash
# 1. Setup environment
cd backend
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train mask detection models
python training.py
# This will:
# - Download Kaggle datasets (~300MB)
# - Process masked/unmasked faces
# - Train SVM/Random Forest models
# - Create PKL files

# 4. Start backend
python main.py

# 5. Start frontend (new terminal)
cd frontend
npm install
npm start

# 6. Register users and test mask recognition!
```

### Files Created After Training
```
backend/
├── face_database.pkl          # Face embeddings with mask data
├── face_recognition_model.pkl # Trained ML models
├── model_info.pkl            # Training metadata
├── users_database.json       # User data (gitignored)
└── recognition_logs.json     # Recognition history (gitignored)
```

## 🎯 Training Details

### What `python training.py` Does:
1. **Downloads Datasets**: 3 mask detection datasets from Kaggle (~300MB total)
2. **Processes Images**: Extracts face embeddings for masked/unmasked faces
3. **Trains Models**: SVM + Random Forest on the processed embeddings
4. **Creates Database**: Mask-aware face database with separate embeddings
5. **Saves PKL Files**: All training data saved for production use

### Training Output:
```
🎭 Face Recognition with Mask Detection - Training System
======================================================================
📋 This will:
   1. Initialize InsightFace buffalo_l model
   2. Download mask detection datasets from Kaggle (~300MB)
   3. Process masked and unmasked face images
   4. Train SVM/Random Forest models
   5. Create mask-aware face database
   6. Save PKL files for production use
======================================================================

🚀 Start training? (y/n): y
```

## 🛠️ Dependencies

### Backend Requirements (`backend/requirements.txt`)
```txt
fastapi>=0.68.0
uvicorn>=0.15.0
insightface>=0.7.3
onnxruntime>=1.10.0
opencv-python>=4.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
Pillow>=8.3.0
pydantic>=1.8.0
python-multipart>=0.0.5
joblib>=1.1.0
kaggle>=1.5.0
```

### Frontend Requirements
- React.js 18+
- Ant Design UI
- Axios for API calls
- React Webcam for camera access

## 🎭 How Mask Detection Works

### Recognition Pipeline:
1. **Face Detection**: InsightFace detects faces in image
2. **Mask Detection**: Color analysis determines if mask is present
3. **Embedding Extraction**: 512-dim face embedding extracted
4. **Smart Matching**: 
   - If masked: Compare with mask embeddings first
   - If unmasked: Compare with regular embeddings
   - Use best similarity score
5. **Adaptive Thresholds**: Lower threshold for masked faces

### Training Data:
- **Masked Faces**: Processed from `with_mask/` directories
- **Unmasked Faces**: Processed from `without_mask/` directories
- **Multiple Identities**: Creates diverse person identities for training
- **Embeddings Storage**: Separate storage for mask/unmask embeddings per person

## 📊 API Endpoints

### Core Recognition
- `POST /api/faces/register` - Register new person
- `POST /api/faces/recognize` - Recognize faces
- `POST /api/faces/live-recognize` - Real-time recognition

### Training & Models
- `POST /api/faces/train-mask-detection` - Train mask detection
- `GET /api/faces/training-status` - Get training progress
- `POST /api/faces/train-models` - Train regular models

### User Management
- `GET /api/faces/users` - List all users
- `DELETE /api/faces/users/{id}` - Delete user
- `POST /api/faces/users/{id}/add-sample` - Add face sample

### Dashboard
- `GET /api/dashboard/stats` - System statistics
- `GET /api/dashboard/recognition-trends` - Recognition trends
- `GET /api/dashboard/top-recognized-users` - Top users

## 🔧 Configuration

### Environment Variables
```bash
# Optional - API configuration
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_API_TIMEOUT=30000

# Kaggle API (for training)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### Model Configuration
- **InsightFace Model**: buffalo_l (highest accuracy)
- **Detection Size**: 640x640 for optimal accuracy
- **Embedding Dimension**: 512
- **Confidence Threshold**: 0.6 (0.48 for masked faces)

## 🚨 Troubleshooting

### Common Issues:

#### Training Fails
```bash
# Check internet connection
ping kaggle.com

# Install/update dependencies
pip install --upgrade insightface onnxruntime

# Run with verbose output
python training.py
```

#### InsightFace Issues
```bash
# CPU-only mode (if GPU issues)
pip uninstall onnxruntime-gpu
pip install onnxruntime

# Clear model cache
rm -rf ~/.insightface/
```

#### Permission Errors
```bash
# Run as administrator (Windows)
# or use sudo (Linux/Mac)
sudo python training.py
```

## 📈 Performance

### System Requirements
- **Minimum**: 4GB RAM, 2 CPU cores, 5GB storage
- **Recommended**: 8GB RAM, 4 CPU cores, 20GB storage
- **GPU**: Optional (NVIDIA GPU with CUDA for faster processing)

### Processing Speed
- **Face Detection**: ~50-100ms per image
- **Recognition**: ~20-50ms per face
- **Training**: ~5-15 minutes (depends on dataset size)

## 🔐 Security & Privacy

- **Local Processing**: All face data stays on your server
- **No Cloud Dependencies**: Works completely offline after training
- **Data Encryption**: Face embeddings are encoded vectors, not raw images
- **User Control**: Users can delete their data anytime

## 📄 License

This project includes training data and model weights that may have their own licenses. Please check the respective model and dataset licenses before commercial use.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 🆘 Support

If you encounter issues:

1. **Check this README** for common solutions
2. **Run training in verbose mode**: `python training.py`
3. **Check logs**: Backend console output and browser developer tools
4. **Verify dependencies**: All packages in requirements.txt installed
5. **Test with simple case**: Register one user without mask first

## 🎯 Next Steps

After setup:
1. **Register yourself** without a mask first
2. **Run mask detection training**: `python training.py` 
3. **Test recognition** with and without mask
4. **Add more users** and test the system
5. **Customize thresholds** in `training.py` if needed

---

**🎭 Ready to recognize faces with masks? Start with `python training.py`!**