# api for the model

import os
import time
from functools import wraps
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status

from src.inference_pipeline import predict_next_n_days

load_dotenv()

from src.utils import load_data_from_s3, load_model_from_s3

app = FastAPI()

def rate_limited(max_call: int, time_frame: int):
    def decorator(func):
        calls = []

        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            now = time.time()
            calls_in_time_frame = [call for call in calls if call > now - time_frame]
            if len(calls_in_time_frame) >= max_call:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests")
            calls.append(now)
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

@app.on_event("startup")
async def startup_event():
    global model, data
    model = load_model_from_s3()
    data = load_data_from_s3()

@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}

@app.get("/")
async def root() -> dict:
    return {"message": "Hello, World!"}

@app.get("/predict/{n}")
@rate_limited(max_call=5, time_frame=60)
async def predict(
    request: Request,
    n: int,
) -> list[dict]:
    """
    Predict the next n days using the model
    Args:
        n: int = 30: number of days to predict
    Returns:
        list[dict]: next n days predictions
    """

    predictions = await predict_next_n_days(
        data=data,
        model=model, 
        n=n,
    ) # list[dict]

    return predictions

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)