from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from app.model.model import predict_image


app = FastAPI()

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    clothing: str

origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins= origins,  # Replace with your React app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/predict", response_model = PredictionOut)
def predict(payload: TextIn):
    clothing = predict_image(payload.text)
    return{"clothing": clothing}