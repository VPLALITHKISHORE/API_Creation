from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from datetime import datetime, timedelta
import asyncio

app = FastAPI()

# CORS setup - allow frontend localhost:3000 or adjust accordingly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load scaler and model once at startup
scaler = joblib.load("scaler.joblib")
model = joblib.load("model.joblib")

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

# Shared state variables
generated_data = []
current_day_index = 0

def generate_synthetic_day_data(day_index: int):
    data = {}
    for key in MEANS:
        data[key] = float(np.random.normal(MEANS[key], STDS[key]))
    date_time = START_DATE + timedelta(days=day_index)
    data["Date_Time"] = date_time.isoformat()
    return data

def detect_anomaly(data_point):
    features = np.array([[data_point["Battery_V"], data_point["Water_Temperature"], data_point["Water_Level"], data_point["Barometric_Pressure"]]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    return "Yes" if prediction == -1 else "No"

async def generate_data_periodically():
    global current_day_index
    while True:
        data_point = generate_synthetic_day_data(current_day_index)
        data_point["Anomaly"] = detect_anomaly(data_point)
        generated_data.append(data_point)
        current_day_index += 1
        await asyncio.sleep(10)  # wait 10 seconds before generating next point

@app.on_event("startup")
async def startup_event():
    # Start background task to generate data every 10 seconds
    asyncio.create_task(generate_data_periodically())

@app.get("/data")
def get_all_data():
    return generated_data

@app.post("/reset")
async def reset_data():
    global generated_data, current_day_index
    generated_data = []
    current_day_index = 0
    return {"message": "Data reset successfully"}
