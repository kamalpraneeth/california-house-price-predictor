import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

model = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    model_path = os.path.join("models", "model.joblib")
    scaler_path = os.path.join("models", "scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully.")
    else:
        print("Warning: Model or scaler not found. Run training script first.")
    yield

app = FastAPI(title="ML Prediction Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: HouseFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
        
    # pydantic v2 syntax
    input_data = pd.DataFrame([features.model_dump()])
    
    try:
        scaled_features = scaler.transform(input_data)
        prediction = model.predict(scaled_features)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for the UI
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
