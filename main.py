from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api import genai_api

app=FastAPI()
app.include_router(genai_api.router)

@app.middleware('http')
async def validate_request(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return JSONResponse(content=str(e), status_code=500)
    