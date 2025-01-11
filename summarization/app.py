from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
model_path = "./models/quantized"
summarizer = pipeline("summarization", model=model_path)


@app.post("/summarize/")
async def summarize_text(text: str):
    result = summarizer(text, max_length=50, min_length=10, do_sample=False)
    return {"summary": result[0]["summary_text"]}
