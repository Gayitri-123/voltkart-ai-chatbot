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
import boto3
import openai
from anthropic import AnthropicBedrockMantle
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
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


NO_AWS_CREDENTIALS = "AWS credentials are not configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."


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
    "claude-opus-4-8": {"label": "Claude Opus 4.8", "provider": "bedrock", "api_model": "anthropic.claude-opus-4-8",
                        "note": "Previous Opus", "effort": "low"},
    "claude-opus-4-7": {"label": "Claude Opus 4.7", "provider": "bedrock", "api_model": "anthropic.claude-opus-4-7",
                        "note": "Previous Opus", "effort": "low"},
    # Non-Claude Bedrock models, called through the Converse API
    "nova-pro": {"label": "Amazon Nova Pro", "provider": "bedrock_converse", "api_model": "us.amazon.nova-pro-v1:0",
                 "note": "Amazon"},
    "nova-2-lite": {"label": "Amazon Nova 2 Lite", "provider": "bedrock_converse", "api_model": "us.amazon.nova-2-lite-v1:0",
                    "note": "Amazon, fast"},
    "llama-4-maverick": {"label": "Llama 4 Maverick", "provider": "bedrock_converse",
                         "api_model": "us.meta.llama4-maverick-17b-instruct-v1:0", "note": "Meta"},
    "mistral-large-3": {"label": "Mistral Large 3", "provider": "bedrock_converse",
                        "api_model": "mistral.mistral-large-3-675b-instruct", "note": "Mistral AI"},
    "deepseek-v3-2": {"label": "DeepSeek V3.2", "provider": "bedrock_converse", "api_model": "deepseek.v3.2",
                      "note": "DeepSeek"},
    "qwen3-next-80b": {"label": "Qwen3 Next 80B", "provider": "bedrock_converse", "api_model": "qwen.qwen3-next-80b-a3b",
                       "note": "Qwen"},
    "gpt-oss-120b": {"label": "GPT-OSS 120B", "provider": "bedrock_converse", "api_model": "openai.gpt-oss-120b-1:0",
                     "note": "OpenAI open-weight"},
    "gpt-6-astra": {"label": "GPT-6 Astra", "provider": "bedrock_converse", "api_model": "us.openai.gpt-6-astra",
                    "note": "OpenAI on Bedrock"},
    "grok-4-6": {"label": "Grok 4.6", "provider": "bedrock_converse", "api_model": "us.xai.grok-4.6", "note": "xAI"},
    "gpt-5": {"label": "GPT-5", "provider": "openai", "api_model": "gpt-5", "note": "OpenAI"},
    "gpt-5-mini": {"label": "GPT-5 mini", "provider": "openai", "api_model": "gpt-5-mini", "note": "OpenAI, faster"},
    "gpt-4.1": {"label": "GPT-4.1", "provider": "openai", "api_model": "gpt-4.1", "note": "OpenAI"},
    "gpt-4o": {"label": "GPT-4o", "provider": "openai", "api_model": "gpt-4o", "note": "OpenAI"},
    "gpt-4o-mini": {"label": "GPT-4o mini", "provider": "openai", "api_model": "gpt-4o-mini", "note": "OpenAI, faster"},
}
DEFAULT_MODEL = "claude-opus-5"

_bedrock_client = None
_converse_client = None
_openai_client = None


def openai_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def list_models():
    return [
        {"id": mid, "label": m["label"], "provider": m["provider"], "note": m["note"],
         "available": m["provider"] != "openai" or openai_configured()}
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


def _run_bedrock(cfg, transcript, steps, usage):
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
    except RuntimeError as e:
        if "credentials" not in str(e).lower():
            raise
        raise ProviderError(503, NO_AWS_CREDENTIALS)

# ---------- Amazon Bedrock Converse (Nova, Llama, Mistral, DeepSeek, ...) ----------

CONVERSE_TOOLS = {"tools": [
    {"toolSpec": {"name": t["name"], "description": t["description"], "inputSchema": {"json": t["input_schema"]}}}
    for t in TOOLS
]}


def _converse():
    global _converse_client
    if _converse_client is None:
        _converse_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _converse_client


def _to_converse_messages(transcript):
    messages = []
    for t in transcript:
        if t["role"] == "assistant":
            for round_ in t.get("steps", []):
                # Converse caps tool-use IDs at 64 chars.
                messages.append({"role": "assistant", "content": [
                    {"toolUse": {"toolUseId": s["id"][-64:], "name": s["name"], "input": s["input"]}} for s in round_]})
                messages.append({"role": "user", "content": [
                    {"toolResult": {"toolUseId": s["id"][-64:], "content": [{"text": json.dumps(s["output"])}]}}
                    for s in round_]})
        # Converse rejects blank text blocks.
        messages.append({"role": t["role"], "content": [{"text": t["text"] or "(no reply)"}]})
    return messages


def _run_converse(cfg, transcript, steps, usage):
    client = _converse()
    messages = _to_converse_messages(transcript)
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.converse(
                modelId=cfg["api_model"],
                system=[{"text": SYSTEM_PROMPT}],
                messages=messages,
                toolConfig=CONVERSE_TOOLS,
                inferenceConfig={"maxTokens": 4096},
            )
            u = response.get("usage", {})
            _add_usage(usage, u.get("inputTokens", 0), u.get("outputTokens", 0),
                       u.get("cacheReadInputTokens", 0), u.get("cacheWriteInputTokens", 0))
            msg = response["output"]["message"]
            messages.append(msg)
            tool_uses = [b["toolUse"] for b in msg["content"] if "toolUse" in b]
            if not tool_uses:
                return "".join(b["text"] for b in msg["content"] if "text" in b).strip()

            results, round_ = [], []
            for call in tool_uses:
                output, is_error = run_tool(call["name"], call["input"])
                round_.append({"id": call["toolUseId"], "name": call["name"], "input": call["input"], "output": output})
                result = {"toolUseId": call["toolUseId"], "content": [{"text": json.dumps(output)}]}
                if is_error:
                    result["status"] = "error"
                results.append({"toolResult": result})
            messages.append({"role": "user", "content": results})
            steps.append(round_)
        return FALLBACK_REPLY
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ThrottlingException":
            raise ProviderError(429, "Bedrock is rate limiting requests, please try again shortly.")
        raise ProviderError(502, f"Bedrock error: {e.response['Error']['Message']}")
    except NoCredentialsError:
        raise ProviderError(503, NO_AWS_CREDENTIALS)
    except BotoCoreError:
        raise ProviderError(503, "Could not reach Amazon Bedrock.")


# ---------- OpenAI (GPT) ----------

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


def _run_openai(cfg, transcript, steps, usage):
    if not openai_configured():
        raise ProviderError(400, "OpenAI models are not configured. Set OPENAI_API_KEY and restart the server.")
    client = _openai()
    messages = _to_openai_messages(transcript)
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
    except openai.RateLimitError as e:
        # OpenAI also returns 429 when the account has run out of credits.
        if e.type == "insufficient_quota":
            raise ProviderError(402, "The OpenAI account has no credits left. Please pick another model.")
        raise ProviderError(429, "OpenAI is rate limiting requests, please try again shortly.")
    except openai.APIStatusError as e:
        raise ProviderError(502, f"OpenAI error: {e.message}")
    except openai.APIConnectionError:
        raise ProviderError(503, "Could not reach OpenAI.")


def _add_usage(usage, input_tokens, output_tokens, cache_read, cache_write):
    """Sum token usage over every model call (tool rounds) behind one reply. input_tokens excludes cached tokens."""
    for key, n in (("input_tokens", input_tokens), ("output_tokens", output_tokens),
                   ("cache_read_tokens", cache_read), ("cache_write_tokens", cache_write)):
        usage[key] = usage.get(key, 0) + (n or 0)


def run_turn(model_id: str, transcript: list, usage: dict | None = None) -> tuple[str, list]:
    """Returns (reply text, tool-call steps made for this reply). Fills usage, if given, with the tokens spent."""
    cfg = MODELS.get(model_id)
    if cfg is None:
        raise ProviderError(400, f"Unknown model {model_id}")
    steps: list = []
    run = {"bedrock": _run_bedrock, "bedrock_converse": _run_converse, "openai": _run_openai}[cfg["provider"]]
    return run(cfg, transcript, steps, {} if usage is None else usage), steps
