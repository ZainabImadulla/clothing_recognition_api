from fastapi import FastAPI
from pydantic import BaseModel

from app.model.model import predict_image


app = FastAPI()

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    clothing: str


@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/predict", response_model = PredictionOut)
def predict(payload: TextIn):
    clothing = predict_image(payload.text)
    return{"clothing": clothing}