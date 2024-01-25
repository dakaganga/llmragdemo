from fastapi import APIRouter
from service import generative_ai_service
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request

router = APIRouter()
templates = Jinja2Templates(directory="templates")
@router.get("/")
def home():
    return "Success"


@router.get("/welcome",response_class=HTMLResponse)
def welcome(request: Request):
    return templates.TemplateResponse(
       name="welcome.html", context = {'request': request, 'data': 75}
        )

'''
@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse(
        request=request, name="item.html", context={"id": id}
    )
'''
@router.get("/genai/")
async def get_answer(query:str):
    return generative_ai_service.query(query)
