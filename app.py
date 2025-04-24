from fastapi import FastAPI
from api import router as api_router

app = FastAPI(
    title="Email Classifier API",
    description="API for classifying emails and masking PII",
    version="1.0.0"
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Email Classifier API!"}

app.include_router(api_router)
