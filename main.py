from fastapi import FastAPI
from src.inference import generate

app = FastAPI()

@app.get("/")
def health():
	return {"message": "Service is working!!!"}

@app.get("/generate")
async def generate():
	pass