from fastapi import FastAPI
from inference_onnx import ONNXPredictor
app = FastAPI(title="Emotion Classifier App")

predictor = ONNXPredictor("./models/model.onnx")

@app.get("/")
async def home_page():
    return "NLP Inference API"


@app.get("/predict")
async def get_prediction(text: str):
    result =  predictor.predict(text)
    return result