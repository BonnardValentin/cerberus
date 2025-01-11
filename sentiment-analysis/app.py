from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
model_path = "./models/quantized"
sentiment_analyzer = pipeline("sentiment-analysis", model=model_path)


@app.post("/analyze-sentiment/")
async def analyze_sentiment(text: str):
    result = sentiment_analyzer(text)
    return {"sentiment": result}
