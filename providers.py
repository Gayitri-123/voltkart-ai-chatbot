"""Model providers. Each provider runs one customer turn (including tool calls) and returns the reply text.

Sessions store a provider-neutral transcript, so customers can switch models mid-conversation:
  {"role": "user", "text": str}
  {"role": "assistant", "text": str, "steps": [[{"id", "name", "input", "output"}, ...], ...]}
"steps" holds the tool calls made for that reply (one inner list per round), replayed in each provider's
native tool-call format so a newly selected model sees the same store data the previous model looked up.
"""

import json
import os

import anthropic
import openai
from anthropic import AnthropicBedrockMantle
from openai import OpenAI

from tools import TOOLS, run_tool

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
STORE_NAME = os.environ.get("STORE_NAME", "VoltKart")
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = f"""You are the customer support assistant for {STORE_NAME}, an Indian consumer electronics retailer.

You help customers with:
- Finding products and understanding specs, prices, offers and stock (use search_products and get_product_details).
- Order status, delivery and refund progress (use get_order_status; ask for the order ID and registered phone number first).
- Returns, refunds, cancellation, delivery, warranty and payment policies (use get_policy).

Only state prices, stock, order details and policies that come from tool results; never invent them. If a tool finds nothing, say so and suggest next steps.
Quote prices in rupees (Rs). Keep replies short and friendly. You may use **bold** for product names and key figures, and "- " bullet lists when listing products or details. Do not use headings or tables.
For anything you cannot resolve (payment disputes, damaged-on-arrival claims, complaints), direct the customer to {STORE_NAME} customer care."""

FALLBACK_REPLY = "Sorry, that took too long to look up. Please try rephrasing your question."


class ProviderError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# id -> display info + how to call it
MODELS = {
    "claude-opus-5": {"label": "Claude Opus 5", "provider": "bedrock", "api_model": "anthropic.claude-opus-5",
                      "note": "Most capable", "effort": "low"},
    "claude-sonnet-5": {"label": "Claude Sonnet 5", "provider": "bedrock", "api_model": "anthropic.claude-sonnet-5",
                        "note": "Balanced", "effort": "low"},
    "claude-haiku-4-5": {"label": "Claude Haiku 4.5", "provider": "bedrock", "api_model": "anthropic.claude-haiku-4-5",
                         "note": "Fastest", "effort": None},
    "gpt-4o": {"label": "GPT-4o", "provider": "openai", "api_model": "gpt-4o", "note": "OpenAI"},
    "gpt-4o-mini": {"label": "GPT-4o mini", "provider": "openai", "api_model": "gpt-4o-mini", "note": "OpenAI, faster"},
}
DEFAULT_MODEL = "claude-opus-5"

_bedrock_client = None
_openai_client = None


def openai_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def list_models():
    return [
        {"id": mid, "label": m["label"], "provider": m["provider"], "note": m["note"],
         "available": m["provider"] == "bedrock" or openai_configured()}
        for mid, m in MODELS.items()
    ]


# ---------- Amazon Bedrock (Claude) ----------

def _bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = AnthropicBedrockMantle(aws_region=AWS_REGION)
    return _bedrock_client


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


def _run_bedrock(cfg, transcript, steps):
    client = _bedrock()
    messages = _to_claude_messages(transcript)
    extra = {"output_config": {"effort": cfg["effort"]}} if cfg["effort"] else {}
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.messages.create(
                model=cfg["api_model"],
                max_tokens=16000,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS,
                messages=messages,
                **extra,
            )
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


# ---------- OpenAI (GPT-4o) ----------

OPENAI_TOOLS = [
    {"type": "function",
     "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in TOOLS
]


def _openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _to_openai_messages(transcript):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in transcript:
        if t["role"] == "assistant":
            for round_ in t.get("steps", []):
                # OpenAI caps tool-call IDs at 40 chars; IDs recorded from Claude turns can be longer.
                messages.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": s["id"][-40:], "type": "function",
                     "function": {"name": s["name"], "arguments": json.dumps(s["input"])}} for s in round_]})
                messages += [{"role": "tool", "tool_call_id": s["id"][-40:], "content": json.dumps(s["output"])} for s in round_]
        messages.append({"role": t["role"], "content": t["text"]})
    return messages


def _run_openai(cfg, transcript, steps):
    if not openai_configured():
        raise ProviderError(400, "GPT-4o is not configured. Set OPENAI_API_KEY and restart the server.")
    client = _openai()
    messages = _to_openai_messages(transcript)
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.chat.completions.create(
                model=cfg["api_model"], messages=messages, tools=OPENAI_TOOLS, max_tokens=4096,
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                return msg.content or ""
            messages.append(msg.model_dump(exclude_none=True))
            round_ = []
            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                    output, _ = run_tool(call.function.name, args)
                except json.JSONDecodeError:
                    args, output = {}, {"error": "Tool arguments were not valid JSON."}
                round_.append({"id": call.id, "name": call.function.name, "input": args, "output": output})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(output)})
            steps.append(round_)
        return FALLBACK_REPLY
    except openai.AuthenticationError:
        raise ProviderError(401, "OpenAI rejected the API key. Check OPENAI_API_KEY.")
    except openai.RateLimitError:
        raise ProviderError(429, "OpenAI is rate limiting requests, please try again shortly.")
    except openai.APIStatusError as e:
        raise ProviderError(502, f"OpenAI error: {e.message}")
    except openai.APIConnectionError:
        raise ProviderError(503, "Could not reach OpenAI.")


def run_turn(model_id: str, transcript: list) -> tuple[str, list]:
    """Returns (reply text, tool-call steps made for this reply)."""
    cfg = MODELS.get(model_id)
    if cfg is None:
        raise ProviderError(400, f"Unknown model {model_id}")
    steps: list = []
    run = _run_bedrock if cfg["provider"] == "bedrock" else _run_openai
    return run(cfg, transcript, steps), steps
