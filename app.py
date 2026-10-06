import os
import time
import uuid

from dotenv import load_dotenv

# Load .env (e.g. OPENAI_API_KEY) before providers reads the environment.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from providers import DEFAULT_MODEL, MODELS, STORE_NAME, ProviderError, list_models, run_turn

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title=f"{STORE_NAME} Assistant")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# session_id -> transcript (format documented in providers.py)
# In-memory; use Redis/DynamoDB in production.
sessions: dict[str, list] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model: str = DEFAULT_MODEL
    debug: bool = False  # testing: also return the tool calls behind the reply


@app.get("/api/models")
def models():
    return {"store_name": STORE_NAME, "default": DEFAULT_MODEL, "models": list_models()}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if req.model not in MODELS:
        raise HTTPException(400, f"Unknown model {req.model}")
    session_id = req.session_id or str(uuid.uuid4())
    transcript = sessions.setdefault(session_id, [])
    pending = transcript + [{"role": "user", "text": req.message}]

    started = time.monotonic()
    try:
        usage: dict = {}
        reply, steps = run_turn(req.model, pending, usage)
    except ProviderError as e:
        raise HTTPException(e.status, str(e))

    # Only record the turn once it succeeded, so a failed call never leaves a dangling user message.
    transcript.extend([pending[-1], {"role": "assistant", "text": reply, "steps": steps}])
    trace = {"tools": [call for round_ in steps for call in round_]} if req.debug else None
    return {
        "session_id": session_id,
        "reply": reply,
        "model": req.model,
        "model_label": MODELS[req.model]["label"],
        "latency_ms": int((time.monotonic() - started) * 1000),
        "usage": usage,  # tokens over every model call behind this reply (tool rounds included)
        **({"trace": trace} if trace else {}),
    }


@app.post("/api/reset")
def reset(req: ChatRequest):
    if req.session_id:
        sessions.pop(req.session_id, None)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
