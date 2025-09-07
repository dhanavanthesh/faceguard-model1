# Minimal Face Recognition Backend

A minimal FastAPI backend based on the Jupyter notebook face recognition system using InsightFace buffalo_l model.

## Quick Start

1. **Install dependencies:**
   ```bash
   cd backend
   python run.py install
   ```

2. **Run training (optional):**
   ```bash
   python run.py train
   ```

3. **Start the API server:**
   ```bash
   python run.py server
   ```

The API will be available at http://localhost:8000

## API Endpoints

- `GET /` - Root endpoint with system info
- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /recognize` - Recognize faces in uploaded image
- `POST /register` - Register new face with name
- `GET /users` - List all registered users
- `DELETE /users/{name}` - Delete a user
- `POST /train` - Trigger model training

## Manual Commands

### Install requirements:
```bash
pip install -r requirements.txt
```

### Run training:
```bash
python training.py
```

### Start server with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Files

- `main.py` - FastAPI application
- `training.py` - Training module (Cells 1 & 2 from notebook)
- `run.py` - Helper script for common tasks
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Notes

- The system automatically loads the InsightFace buffalo_l model on startup
- Training data is loaded from `mask_dataset1` and `mask_dataset2` directories if available
- Face database is saved to `face_database.pkl`
- Trained models are saved to `face_recognition_model.pkl`