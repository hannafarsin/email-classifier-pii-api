from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils import process_email

router = APIRouter()

class EmailRequest(BaseModel):
    subject: str
    body: str

@router.post("/classify")
async def classify_email(email: EmailRequest):
    try:
        response = process_email(email.subject, email.body)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing email: {str(e)}")
