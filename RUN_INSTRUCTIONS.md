# 🚀 How to Run Face Recognition System

## Quick Start with Virtual Environment

### Terminal 1: Backend Setup
```bash
cd backend

# Option 1: Auto setup venv + install
python setup_venv.py setup

# Option 2: Manual setup  
python setup_venv.py create
# Windows: face_recognition_env\Scripts\activate
# Linux/Mac: source face_recognition_env/bin/activate
pip install -r requirements.txt

# Optional: Run training (creates face database)
python training.py

# Run backend
python main.py
```
Backend runs at: http://localhost:8000

### Terminal 2: Frontend  
```bash
cd frontend
npm install
npm start
```
Frontend runs at: http://localhost:3000

## ✅ System Ready
- Frontend connects to backend automatically
- CORS configured for port 3000
- API endpoints match frontend expectations

## 📋 API Endpoints (Backend)

### Core Endpoints:
- `GET /health` - Health check
- `POST /api/faces/register-file` - Register new person  
- `POST /api/faces/recognize-file` - Recognize faces
- `GET /api/faces/users` - List users
- `DELETE /api/faces/users/{name}` - Delete user
- `POST /api/faces/train-models` - Train models

### Dashboard:
- `GET /api/dashboard/stats` - System statistics
- `GET /api/dashboard/system-performance` - Performance metrics

## 🎯 Training Workflow

### Step 1: Run Training (Required for best accuracy)
```bash
cd backend
# Activate venv first (if using venv)
python training.py
```

**What training does:**
- Loads InsightFace buffalo_l model (Cell 1)
- Downloads/processes datasets if available (Cell 2)  
- Trains SVM + Random Forest models
- Creates face database with embeddings
- Saves models to disk

### Step 2: After Training - System Works Correctly
✅ **Face database** created with known identities  
✅ **ML models** trained for accurate recognition  
✅ **Fallback** cosine similarity for unknown faces  
✅ **Registration** system for new faces  

**Files created after training:**
- `face_database.pkl` - Face embeddings database
- `face_recognition_model.pkl` - Trained ML model  
- `model_info.pkl` - Model metadata

### Manual Training Trigger (via API)
```bash
curl -X POST http://localhost:8000/api/faces/train-models
```

## 🌐 Access Points
- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000  
- **API Docs**: http://localhost:8000/docs

## 📱 Usage
1. Open http://localhost:3000 in browser
2. Register faces using the Registration tab
3. Test recognition using Live Recognition tab
4. View statistics in Dashboard tab

That's it! The system is ready to use.