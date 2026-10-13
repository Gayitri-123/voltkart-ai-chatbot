"""Model providers. Each provider runs one customer turn (including tool calls) and returns the reply text.

Sessions store a provider-neutral transcript, so customers can switch models mid-conversation:
  {"role": "user", "text": str}
  {"role": "assistant", "text": str, "steps": [[{"id", "name", "input", "output"}, ...], ...]}
"steps" holds the tool calls made for that reply (one inner list per round), replayed in each provider's
native tool-call format so a newly selected model sees the same store data the previous model looked up.

Three sources are offered: Google Gemini (Gemini API free tier) and open-weights models served locally by Ollama, both
through their OpenAI-compatible APIs, and Anthropic Claude (on Amazon Bedrock, billed to the AWS account, through the
Anthropic SDK).
"""

import json
import os

import anthropic
import boto3
import httpx
import openai
from anthropic import AnthropicBedrockMantle
from openai import OpenAI

from tools import TOOLS, run_tool

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
STORE_NAME = os.environ.get("STORE_NAME", "VoltKart")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
# Local models can be slow, especially the first call while Ollama loads the model into memory.
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = f"""You are the customer support assistant for {STORE_NAME}, an Indian consumer electronics retailer.

You help customers with:
- Finding products and understanding specs, prices, offers and stock (use search_products and get_product_details).
- Order status, delivery and refund progress (use get_order_status; ask for the order ID and registered phone number first).
- Returns, refunds, cancellation, delivery, warranty and payment policies (use get_policy).

How to answer:
- Look things up before answering. For any product question, including broad ones (a budget, "the cheapest", "what do you sell", a product named loosely or misspelt), call search_products first: with no query, or only max_price or category, when the request is broad. The catalog is small, so list the matching products with their prices instead of asking which category the customer wants. Ask a clarifying question only if the search finds nothing relevant.
- Only state prices, stock, specs, offers, order details and policies that come from tool results; never invent them. Don't fill gaps with general knowledge ("most models have...", "typically 1-2 years"): if a detail isn't in the data, say you don't have it. If a tool finds nothing, say so and suggest next steps.
- Treat every customer the same. Give the same facts, offers (EMI, bank discounts), warranty details and depth of answer whatever their name, gender, religion, caste, nationality, age, disability, job or income, and however they write (Hinglish, informal, typos). Never comment on whether they can afford or need a product, and don't go along with stereotypes.
- Stay on topic: politely decline homework, legal, medical, investment and other requests unrelated to {STORE_NAME} shopping and support, and offer shopping help instead.
- Your data has no customer-care phone numbers, emails, chat channels or hours. For anything you cannot resolve (payment disputes, damaged-on-arrival claims, complaints, a request for a human), say {STORE_NAME} customer care will need to help, without inventing how to reach them or any process, timeline or compensation. For damaged or defective products, also give the returns policy from get_policy.

Quote prices in rupees (Rs). Keep replies short and friendly. You may use **bold** for product names and key figures, and "- " bullet lists when listing products or details. Do not use headings or tables."""

FALLBACK_REPLY = "Sorry, that took too long to look up. Please try rephrasing your question."

NO_AWS_CREDENTIALS = "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY on the server."


class ProviderError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _aws_configured():
    return boto3.Session().get_credentials() is not None


# provider -> how to reach it. "configured": whether its models can be offered; "missing": what to set if not.
PROVIDERS = {
    # Free key: https://aistudio.google.com/apikey
    "gemini": {"name": "Gemini", "configured": lambda: bool(os.environ.get("GEMINI_API_KEY")),
               "missing": "Set GEMINI_API_KEY on the server."},
    # Paid: AWS credentials from the environment or ~/.aws, with Bedrock access to the Claude models.
    "bedrock": {"name": "Amazon Bedrock", "configured": _aws_configured, "missing": NO_AWS_CREDENTIALS},
    # Free, local: open-weights models pulled into Ollama (https://ollama.com). Its models are found at runtime.
    "ollama": {"name": "Ollama", "configured": lambda: bool(_ollama),
               "missing": f"Start Ollama ({OLLAMA_URL}) and pull a model that supports tools, e.g. `ollama pull qwen3`."},
}

_NOTES = {"gemini": "Free tier, Gemini API", "bedrock": "Paid, Amazon Bedrock", "ollama": "Free, local (Ollama)"}

# (id, label, provider, model name in that provider's API). Only models that support tool calling.
_MODEL_LIST = [
    ("gemini-3.8-flash", "Gemini 3.8 Flash", "gemini", "gemini-3.8-flash"),
    ("gemini-3.7-flash", "Gemini 3.7 Flash", "gemini", "gemini-3.7-flash"),
    ("gemini-3.6-flash", "Gemini 3.6 Flash", "gemini", "gemini-3.6-flash"),
    ("gemini-3.5-flash", "Gemini 3.5 Flash", "gemini", "gemini-3.5-flash"),
    ("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", "gemini", "gemini-3.5-flash-lite"),
    ("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", "gemini", "gemini-3.1-flash-lite"),
    ("gemini-3-flash-preview", "Gemini 3 Flash Preview", "gemini", "gemini-3-flash-preview"),
    ("gemini-2.5-flash", "Gemini 2.5 Flash", "gemini", "gemini-2.5-flash"),
    ("claude-opus-5", "Claude Opus 5", "bedrock", "anthropic.claude-opus-5"),
    ("claude-sonnet-5", "Claude Sonnet 5", "bedrock", "anthropic.claude-sonnet-5"),
    ("claude-haiku-4-5", "Claude Haiku 4.5", "bedrock", "anthropic.claude-haiku-4-5"),
]
MODELS = {mid: {"label": label, "provider": provider, "api_model": api_model, "note": _NOTES[provider]}
          for mid, label, provider, api_model in _MODEL_LIST}
DEFAULT_MODEL = "gemini-3.8-flash"

# Gemini 3 needs the thought signature of each of its tool calls sent back. Tool calls replayed from another model
# (or saved without one) get this placeholder, which Gemini accepts in place of a real signature.
DUMMY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"

_clients = {}

_ollama = {}  # id -> model info for the Ollama models pulled on this machine, refreshed by _refresh_ollama()
_ollama_tools = {}  # Ollama model digest -> whether that model supports tool calling


def _available(provider):
    return PROVIDERS[provider]["configured"]()


def _not_configured(provider):
    p = PROVIDERS[provider]
    return ProviderError(503, f"{p['name']} is not configured. {p['missing']}")


def _ollama_models():
    """Pulled Ollama models that support tool calling (entries from /api/tags). Empty if Ollama isn't running."""
    try:
        found = []
        for m in httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2).json()["models"]:
            if m["digest"] not in _ollama_tools:
                show = httpx.post(f"{OLLAMA_URL}/api/show", json={"model": m["name"]}, timeout=10).json()
                _ollama_tools[m["digest"]] = "tools" in show.get("capabilities", [])
            if _ollama_tools[m["digest"]]:
                found.append(m)
        return found
    except (httpx.HTTPError, ValueError, KeyError):
        return []


def _refresh_ollama():
    global _ollama
    models = {}
    for m in _ollama_models():
        size = (m.get("details") or {}).get("parameter_size")
        models[f"ollama/{m['name']}"] = {"label": m["name"], "provider": "ollama", "api_model": m["name"],
                                         "note": f"{_NOTES['ollama']}, {size}" if size else _NOTES["ollama"]}
    _ollama = models  # replaced whole, so requests reading it on other threads never see it half-built


def get_model(model_id):
    """Model info for an id, or None if unknown."""
    if model_id.startswith("ollama/") and model_id not in _ollama:
        _refresh_ollama()  # pulled since the model list was last loaded
    return MODELS.get(model_id) or _ollama.get(model_id)


def list_models():
    _refresh_ollama()
    available = {p: _available(p) for p in PROVIDERS}
    return [{"id": mid, "label": m["label"], "provider": m["provider"], "note": m["note"],
             "available": available[m["provider"]]}
            for mid, m in {**MODELS, **_ollama}.items()]


# ---------- Gemini and Ollama (OpenAI-compatible APIs) ----------

OPENAI_TOOLS = [
    {"type": "function",
     "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOLS
]


def _client(provider):
    # max_retries: the free tier rate-limits quickly; the SDK backs off and retries 429s.
    if provider not in _clients:
        if provider == "ollama":
            _clients[provider] = OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama", max_retries=0,
                                        timeout=OLLAMA_TIMEOUT)
        else:
            _clients[provider] = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                                        api_key=os.environ["GEMINI_API_KEY"], max_retries=2, timeout=60)
    return _clients[provider]


def _replayed_call(step, provider):
    # Tool-call IDs are capped at 40 chars; IDs recorded from Claude turns can be longer.
    call = {"id": step["id"][-40:], "type": "function",
            "function": {"name": step["name"], "arguments": json.dumps(step["input"])}}
    if provider == "gemini":
        call["extra_content"] = {"google": {"thought_signature": step.get("thought_signature") or DUMMY_THOUGHT_SIGNATURE}}
    return call


def _to_openai_messages(transcript, provider):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in transcript:
        if t["role"] == "assistant":
            for round_ in t.get("steps", []):
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [_replayed_call(s, provider) for s in round_]})
                messages += [{"role": "tool", "tool_call_id": s["id"][-40:], "content": json.dumps(s["output"])} for s in round_]
        messages.append({"role": t["role"], "content": t["text"]})
    return messages


def _thought_signature(call):
    """Gemini 3 returns a thought signature with each tool call, under extra_content.google."""
    extra = (call.model_extra or {}).get("extra_content") or {}
    return (extra.get("google") or {}).get("thought_signature")


def _run_openai(cfg, transcript, steps, usage):
    provider = cfg["provider"]
    if not _available(provider):
        raise _not_configured(provider)
    client = _client(provider)
    messages = _to_openai_messages(transcript, provider)
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.chat.completions.create(
                model=cfg["api_model"], messages=messages, tools=OPENAI_TOOLS, max_completion_tokens=16000,
            )
            if response.usage:
                cached = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0
                _add_usage(usage, response.usage.prompt_tokens - cached, response.usage.completion_tokens, cached, 0)
            msg = response.choices[0].message
            if not msg.tool_calls:
                return msg.content or ""
            messages.append(msg.model_dump(exclude_none=True))  # keeps Gemini's extra_content (thought signatures)
            round_ = []
            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                    output, _ = run_tool(call.function.name, args)
                except json.JSONDecodeError:
                    args, output = {}, {"error": "Tool arguments were not valid JSON."}
                step = {"id": call.id, "name": call.function.name, "input": args, "output": output}
                if signature := _thought_signature(call):
                    step["thought_signature"] = signature
                round_.append(step)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(output)})
            steps.append(round_)
        return FALLBACK_REPLY
    except openai.RateLimitError as e:
        # Gemini names the quota it hit: a daily one (GenerateRequestsPerDay...) won't clear by waiting a minute.
        if "perday" in f"{e.message} {e.body}".lower().replace(" ", ""):
            raise ProviderError(429, f"{cfg['label']} has used up its free daily limit. Pick another model, "
                                     "or try this one again tomorrow.")
        raise ProviderError(429, f"{cfg['label']} hit its free-tier limit. Wait a minute or pick another model.")
    except openai.AuthenticationError:
        raise ProviderError(502, "Gemini rejected the API key. Check GEMINI_API_KEY.")
    except openai.NotFoundError:
        if provider == "ollama":
            raise ProviderError(502, f"{cfg['api_model']} is not pulled in Ollama. Run `ollama pull {cfg['api_model']}`.")
        raise ProviderError(502, f"Gemini does not offer {cfg['api_model']} to this account.")
    except openai.APIStatusError as e:
        raise ProviderError(502, f"{PROVIDERS[provider]['name']} error: {e.message}")
    except openai.APITimeoutError:
        raise ProviderError(504, f"{cfg['label']} took too long to reply.")
    except openai.APIConnectionError:
        if provider == "ollama":
            raise ProviderError(503, f"Could not reach Ollama at {OLLAMA_URL}. Is it running (`ollama serve`)?")
        raise ProviderError(503, "Could not reach Gemini.")


# ---------- Claude (Amazon Bedrock) ----------

def _bedrock():
    if "bedrock" not in _clients:
        _clients["bedrock"] = AnthropicBedrockMantle(aws_region=AWS_REGION)
    return _clients["bedrock"]


def _to_claude_messages(transcript):
    messages = []
    for t in transcript:
        if t["role"] == "assistant":
            for round_ in t.get("steps", []):
                messages.append({"role": "assistant", "content": [
                    {"type": "tool_use", "id": s["id"], "name": s["name"], "input": s["input"]} for s in round_]})
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": s["id"], "content": json.dumps(s["output"])} for s in round_]})
        messages.append({"role": t["role"], "content": t["text"]})
    return messages


def _run_bedrock(cfg, transcript, steps, usage):
    if not _available("bedrock"):
        raise _not_configured("bedrock")
    client = _bedrock()
    messages = _to_claude_messages(transcript)
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.messages.create(
                model=cfg["api_model"],
                max_tokens=16000,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS,
                messages=messages,
            )
            u = response.usage
            _add_usage(usage, u.input_tokens, u.output_tokens,
                       u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0)
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "refusal":
                return f"Sorry, I can't help with that. Please contact {STORE_NAME} customer care."
            if response.stop_reason != "tool_use":
                return "".join(b.text for b in response.content if b.type == "text")

            results, round_ = [], []
            for block in response.content:
                if block.type == "tool_use":
                    output, is_error = run_tool(block.name, block.input)
                    round_.append({"id": block.id, "name": block.name, "input": block.input, "output": output})
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(output), "is_error": is_error})
            messages.append({"role": "user", "content": results})
            steps.append(round_)
        return FALLBACK_REPLY
    except anthropic.RateLimitError:
        raise ProviderError(429, "Bedrock is rate limiting requests, please try again shortly.")
    except anthropic.APIStatusError as e:
        raise ProviderError(502, f"Bedrock error: {e.message}")
    except anthropic.APIConnectionError:
        raise ProviderError(503, "Could not reach Amazon Bedrock.")


def _add_usage(usage, input_tokens, output_tokens, cache_read, cache_write):
    """Sum token usage over every model call (tool rounds) behind one reply. input_tokens excludes cached tokens."""
    for key, n in (("input_tokens", input_tokens), ("output_tokens", output_tokens),
                   ("cache_read_tokens", cache_read), ("cache_write_tokens", cache_write)):
        usage[key] = usage.get(key, 0) + (n or 0)


def run_turn(model_id: str, transcript: list, usage: dict | None = None) -> tuple[str, list]:
    """Returns (reply text, tool-call steps made for this reply). Fills usage, if given, with the tokens spent."""
    cfg = get_model(model_id)
    if cfg is None:
        raise ProviderError(400, f"Unknown model {model_id}")
    steps: list = []
    run = _run_bedrock if cfg["provider"] == "bedrock" else _run_openai
    return run(cfg, transcript, steps, {} if usage is None else usage), steps


def complete(model_id: str, prompt: str) -> str:
    """One plain completion (no tools), e.g. for the eval judge."""
    cfg = get_model(model_id)
    if cfg["provider"] == "bedrock":
        response = _bedrock().messages.create(model=cfg["api_model"], max_tokens=4000,
                                              messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in response.content if b.type == "text")
    response = _client(cfg["provider"]).chat.completions.create(
        model=cfg["api_model"], messages=[{"role": "user", "content": prompt}], max_completion_tokens=4000)
    return response.choices[0].message.content or ""
