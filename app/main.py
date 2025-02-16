from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model.model import predict_pipeline

app = FastAPI()
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace '*' with specific domains in production, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    language: str

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    try:
        language = predict_pipeline(payload.text)
        return {"language": language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
