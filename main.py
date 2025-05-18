from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import asyncio

app = FastAPI()

# CORS setup (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup
scaler = joblib.load("scaler.joblib")
model = joblib.load("model.joblib")

# Feature statistics (mean, std)
MEANS = {
    "Battery_V": 3.625691,
    "Water_Temperature": 26.393680,
    "Water_Level": -5.996789,
    "Barometric_Pressure": 952.831166,
}

STDS = {
    "Battery_V": 0.049711,
    "Water_Temperature": 0.155034,
    "Water_Level": 4.034594,
    "Barometric_Pressure": 5.905373,
}

START_DATE = datetime(2024, 10, 9)

# Shared state
generated_data = []
current_day_index = 0
task_started = False  # Prevent multiple background tasks

# Generate one synthetic data point
def generate_synthetic_day_data(day_index: int):
    data = {}
    for key in MEANS:
        data[key] = float(np.random.normal(MEANS[key], STDS[key]))
    date_time = START_DATE + timedelta(days=day_index)
    data["Date_Time"] = date_time.isoformat()
    return data

# Predict anomaly
def detect_anomaly(data_point):
    df = pd.DataFrame([{
        "Battery_V": data_point["Battery_V"],
        "Water_Temperature": data_point["Water_Temperature"],
        "Water_Level": data_point["Water_Level"],
        "Barometric_Pressure": data_point["Barometric_Pressure"],
    }])
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    return "Yes" if prediction == -1 else "No"

# Background task to generate data every 20 seconds
async def generate_data_periodically():
    global current_day_index
    while True:
        data_point = generate_synthetic_day_data(current_day_index)
        data_point["Anomaly"] = detect_anomaly(data_point)
        generated_data.append(data_point)
        print(f"[{datetime.now()}] Added data for day {current_day_index}: {data_point}")
        current_day_index += 1
        await asyncio.sleep(20)  # Wait 20 seconds before generating next

# Startup: launch background generator once
@app.on_event("startup")
async def startup_event():
    global task_started
    if not task_started:
        asyncio.create_task(generate_data_periodically())
        task_started = True

# Endpoint: get all data
@app.get("/data")
def get_all_data():
    return generated_data

# Endpoint: reset data
@app.post("/reset")
async def reset_data():
    global generated_data, current_day_index
    generated_data = []
    current_day_index = 0
    return {"message": "Data reset successfully"}
