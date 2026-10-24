from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="ShikshaSetu API")

# Allow Frontend to communicate with Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'dropout_model.pkl')
model = joblib.load(model_path)

# Define Expected Input
class StudentData(BaseModel):
    Gender: str
    Region: str
    Distance_km: float
    No_of_Siblings: int
    Family_Income: str
    Father_Education: str
    Mother_Education: str
    Internet_Access: str
    Stream: str
    Class_10_Percentage: float
    Class_11_Percentage: float
    Attendance_Percentage: float

@app.get("/")
def home():
    return {"message": "ShikshaSetu API is running!"}

@app.post("/predict")
def predict_dropout(data: StudentData):
    student_dict = data.dict()
    df = pd.DataFrame([student_dict])
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1] 
    
    if probability > 0.65:
        risk = "High Risk"
    elif probability > 0.40:
        risk = "Medium Risk"
    else:
        risk = "Low Risk / Safe"

    return {
        "prediction": int(prediction),
        "dropout_probability": round(probability * 100, 2),
        "risk_status": risk,
    }