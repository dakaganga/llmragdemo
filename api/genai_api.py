from fastapi import APIRouter
from service import generative_ai_service

router = APIRouter()

@router.get("/")
def home():
    return "Success"

@router.get("/genai/{query}")
async def get_answer(query):
    return generative_ai_service.query(query)
