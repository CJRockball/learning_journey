from fastapi import FastAPI, Request, Form
import pathlib
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse, RedirectResponse, HTMLResponse
from typing import Optional
from app.BS_pricing import bs_call, bs_put 
import starlette.status as status

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "templates"

app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATE_PATH)


@app.get("/")
def home():
    return {"home":"page"}

@app.get("/bs")
def bs_pred(request: Request, response_class=HTMLResponse):
    dt =365
    return templates.TemplateResponse("bs.html", {"request": request})    

@app.post("/bs")
def bs_pred(request: Request,
            S0: Optional[float] = Form(None), 
            K: Optional[float] = Form(None), 
            r: Optional[float] = Form(None), 
            sigma: Optional[float] = Form(None), 
            T: Optional[int] = Form(None), 
            submit: str = Form(...)):
    
    if submit == "Submit":
        dt =365
        call = round(bs_call(S0, K, T, r, sigma),4)
        put = round(bs_put(S0, K, T, r, sigma),4)        
        #print(calls, put)
        return templates.TemplateResponse("bsp.html", {"request": request,
                                                   "S0":S0,
                                                   "K":K,
                                                   "r":r,
                                                   "sigma":sigma,
                                                   "T":T,
                                                   "call":call,
                                                   "put":put})   
    elif submit == "Reset":
        forw_url = f'/bs' 
        return RedirectResponse(forw_url, status_code=status.HTTP_302_FOUND)


