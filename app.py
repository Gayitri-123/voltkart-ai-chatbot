import hmac
import logging
import os
import re
import secrets
import time

from dotenv import load_dotenv

# Load .env (e.g. GEMINI_API_KEY) before providers reads the environment.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import Cookie, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sessions
from providers import DEFAULT_MODEL, STORE_NAME, ProviderError, get_model, list_models, run_turn

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title=f"{STORE_NAME} Assistant")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Operators send this in the X-Debug-Token header to get tool traces; debug traces are off when it isn't set.
DEBUG_TOKEN = os.environ.get("DEBUG_TOKEN", "")
# Each browser gets a random owner id in this cookie; a conversation can only be used or reset by the browser that
# started it, even if its session id leaks.
OWNER_COOKIE = "chat_owner"

# Tool arguments masked in debug traces, and phone numbers / emails masked anywhere else in them.
SENSITIVE_KEYS = {"phone", "email", "address", "name_on_card", "card_number"}
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?\d{6}(\d{4})(?!\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


# Conversations: in Redis when REDIS_URL is set, otherwise in this process's memory (see sessions.py).
store = sessions.store_from_env()
log = logging.getLogger(__name__)


def store_unavailable(request, exc):
    log.error("Session store error: %r", exc)
    return JSONResponse({"detail": "Chat history is temporarily unavailable. Please try again shortly."}, 503)


for error in sessions.ERRORS:
    app.add_exception_handler(error, store_unavailable)


# Which kind of lookup each tool is; a reply's "route" names the kind it started with (gray-box testing).
TOOL_ROUTES = {"search_products": "product", "get_product_details": "product",
               "get_order_status": "order", "get_policy": "policy"}


def route_of(transcript):
    """The route of the latest turn that called a tool, from its first call (how the model read the request; later
    calls support it, e.g. the cancellation policy after the order). A follow-up that answers from earlier lookups
    stays on that route; "chat" when no turn has looked anything up."""
    for turn in reversed(transcript):
        calls = [call for round_ in turn.get("steps", []) for call in round_]
        if calls:
            return TOOL_ROUTES.get(calls[0]["name"], "chat")
    return "chat"


def redact(value, key=None):
    """A copy of a tool call's input/output fit for a debug trace: sensitive fields and phone numbers keep only their
    last 4 digits, emails are hidden."""
    if isinstance(value, dict):
        return {k: redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if key in SENSITIVE_KEYS and value not in (None, ""):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return f"***{digits[-4:]}" if len(digits) >= 4 else "***"
    if isinstance(value, str):
        return EMAIL_RE.sub("***@***", PHONE_RE.sub(r"******\1", value))
    return value


def debug_allowed(token):
    return bool(DEBUG_TOKEN) and token is not None and hmac.compare_digest(token, DEBUG_TOKEN)


def owner_id(request: Request, response: Response, owner):
    """The caller's owner id from its cookie, issuing a new one when it has none."""
    if owner:
        return owner
    owner = secrets.token_urlsafe(32)
    response.set_cookie(OWNER_COOKIE, owner, httponly=True, samesite="lax", secure=request.url.scheme == "https")
    return owner


def check_owner(session_id, owner):
    """404 unless this owner started the conversation. Unknown, expired and other owners' conversations look the same,
    so ids can't be probed."""
    started_by = store.owner(session_id)
    if started_by is None or not hmac.compare_digest(started_by, owner):
        raise HTTPException(404, "This conversation is no longer available. Please start a new chat.")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model: str = DEFAULT_MODEL
    debug: bool = False  # testing: also return the tool calls behind the reply (needs the X-Debug-Token header)


@app.get("/api/models")
def models():
    return {"store_name": STORE_NAME, "default": DEFAULT_MODEL, "models": list_models()}


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request, response: Response,
         owner: str | None = Cookie(None, alias=OWNER_COOKIE), x_debug_token: str | None = Header(None)):
    model = get_model(req.model)
    if model is None:
        raise HTTPException(400, f"Unknown model {req.model}")
    owner = owner_id(request, response, owner)
    if req.session_id:
        session_id = req.session_id
        check_owner(session_id, owner)
    else:
        session_id = store.create(owner)

    token = store.lock(session_id)
    if token is None:
        raise HTTPException(409, "Still answering your previous message. Please wait for it before sending another.")
    try:
        pending = store.transcript(session_id) + [{"role": "user", "text": req.message}]
        started = time.monotonic()
        try:
            usage: dict = {}
            reply, steps = run_turn(req.model, pending, usage)
        except ProviderError as e:
            raise HTTPException(e.status, str(e))

        # Only record the turn once it succeeded, so a failed call never leaves a dangling user message. It isn't
        # recorded if the conversation was reset meanwhile or the reply outlived its lock; the reply is still sent.
        turns = [pending[-1], {"role": "assistant", "text": reply, "steps": steps}]
        if not store.append(session_id, token, turns):
            log.warning("Reply not recorded: conversation reset or lock expired")
        route = route_of(pending[:-1] + turns)
    finally:
        store.unlock(session_id, token)

    tools = [redact(call) for round_ in steps for call in round_]
    trace = {"tools": tools} if req.debug and debug_allowed(x_debug_token) else None
    return {
        "session_id": session_id,
        "reply": reply,
        "model": req.model,
        "model_label": model["label"],
        "latency_ms": int((time.monotonic() - started) * 1000),
        "usage": usage,  # tokens over every model call behind this reply (tool rounds included)
        "route": route,  # product / order / policy / chat
        **({"trace": trace} if trace else {}),
    }


@app.post("/api/reset")
def reset(req: ChatRequest, owner: str | None = Cookie(None, alias=OWNER_COOKIE)):
    if req.session_id:
        check_owner(req.session_id, owner or "")
        store.delete(req.session_id)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
