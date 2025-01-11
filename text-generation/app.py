from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
model_path = "./models/quantized"
text_generator = pipeline("text-generation", model=model_path)


@app.post("/generate-text/")
async def generate_text(text: str):
    result = text_generator(text, max_length=50, num_return_sequences=1)
    return {"generated_text": result[0]["generated_text"]}
