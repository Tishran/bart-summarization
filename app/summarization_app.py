from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, BartTokenizerFast
import torch
import uvicorn
import os

enabled_device = "cuda" if torch.cuda.is_available() else "cpu"

class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 128
    num_beams: int = 4


MODEL_DIR = os.getenv("MODEL_DIR", "./model")
DEVICE = os.getenv("DEVICE", enabled_device)

tokenizer = BartTokenizerFast.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

app = FastAPI(title="BART Summarization Service")

app.mount("/static", StaticFiles(directory="./static"), name="static")


@app.get("/", response_class=HTMLResponse)
def read_root():
    return FileResponse("./static/index.html")


@app.post("/summarize", response_class=JSONResponse)
def summarize(req: SummarizationRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")
    inputs = tokenizer(
        req.text, return_tensors="pt", truncation=True, max_length=1024
    ).to(DEVICE)
    generated_ids = model.generate(
        **inputs,
        max_length=req.max_length,
        num_beams=req.num_beams,
        length_penalty=2.0,
        early_stopping=True
    )
    summary = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {"summary": summary}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "fastapi_summarization_app:app", host="0.0.0.0", port=port, reload=True
    )
