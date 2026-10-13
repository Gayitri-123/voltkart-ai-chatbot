import json

import anthropic
import httpx
import openai
import pytest

import providers
from conftest import claude_message, completion, tool_call
from providers import ProviderError, run_turn


def _status_error(cls, status):
    request = httpx.Request("POST", "https://example.test")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


def test_model_registry():
    assert providers.DEFAULT_MODEL in providers.MODELS
    for mid, m in providers.MODELS.items():
        assert m["provider"] in providers.PROVIDERS, mid
    assert {m["provider"] for m in providers.MODELS.values()} == {"gemini", "bedrock"}
    assert len(providers.OPENAI_TOOLS) == len(providers.TOOLS)


def test_list_models_availability(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setitem(providers.PROVIDERS["bedrock"], "configured", lambda: True)
    avail = {m["id"]: m["available"] for m in providers.list_models()}
    assert avail["gemini-3.8-flash"] is False
    assert avail["claude-sonnet-5"] is True


def test_plain_reply_and_usage(fake_client):
    client = fake_client([completion("Hello!", prompt_tokens=100, completion_tokens=7, cached=40)])
    usage = {}
    reply, steps = run_turn("gemini-2.5-flash", [{"role": "user", "text": "hi"}], usage)
    assert (reply, steps) == ("Hello!", [])
    assert usage == {"input_tokens": 60, "output_tokens": 7, "cache_read_tokens": 40, "cache_write_tokens": 0}
    req = client.requests[0]
    assert req["model"] == "gemini-2.5-flash"
    assert req["messages"][0]["role"] == "system"
    assert req["messages"][-1] == {"role": "user", "content": "hi"}


def test_tool_loop_runs_tools_and_records_steps(fake_client):
    call = tool_call("get_order_status", json.dumps({"order_id": "ORD10002", "phone": "9876543210"}), signature="sig")
    client = fake_client([completion(tool_calls=[call]), completion("It's out for delivery.")])
    usage = {}
    reply, steps = run_turn("gemini-3.8-flash", [{"role": "user", "text": "where is ORD10002"}], usage)
    assert reply == "It's out for delivery."
    assert len(steps) == 1 and steps[0][0]["name"] == "get_order_status"
    assert steps[0][0]["output"]["status"] == "Out for delivery"
    assert steps[0][0]["thought_signature"] == "sig"
    tool_msg = client.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["status"] == "Out for delivery"
    assert usage["input_tokens"] == 20  # summed over both rounds


def test_invalid_tool_json(fake_client):
    fake_client([completion(tool_calls=[tool_call("get_policy", "{not json")]), completion("ok")])
    _, steps = run_turn("gemini-3.5-flash", [{"role": "user", "text": "x"}])
    assert steps[0][0]["output"] == {"error": "Tool arguments were not valid JSON."}


def test_too_many_tool_rounds_falls_back(fake_client):
    call = tool_call("get_policy", json.dumps({"topic": "returns"}))
    fake_client([completion(tool_calls=[call])] * providers.MAX_TOOL_ROUNDS)
    reply, steps = run_turn("gemini-3.5-flash", [{"role": "user", "text": "x"}])
    assert reply == providers.FALLBACK_REPLY
    assert len(steps) == providers.MAX_TOOL_ROUNDS


def test_claude_tool_loop(fake_bedrock):
    client = fake_bedrock([
        claude_message(tool_uses=[("toolu_1", "get_order_status", {"order_id": "ORD10002", "phone": "9876543210"})]),
        claude_message("It's out for delivery."),
    ])
    usage = {}
    reply, steps = run_turn("claude-sonnet-5", [{"role": "user", "text": "where is ORD10002"}], usage)
    assert reply == "It's out for delivery."
    assert steps[0][0]["id"] == "toolu_1" and steps[0][0]["output"]["status"] == "Out for delivery"
    req = client.requests[1]
    assert req["model"] == "anthropic.claude-sonnet-5"
    result = req["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_1"
    assert usage["input_tokens"] == 20 and usage["output_tokens"] == 10


def test_claude_refusal(fake_bedrock):
    fake_bedrock([claude_message("", stop_reason="refusal")])
    assert "customer care" in run_turn("claude-haiku-4-5", [{"role": "user", "text": "x"}])[0]


def test_history_replay_across_providers():
    step = {"id": "x" * 60, "name": "get_policy", "input": {"topic": "returns"}, "output": {"policy": "7 days"}}
    transcript = [{"role": "user", "text": "returns?"},
                  {"role": "assistant", "text": "7 days.", "steps": [[step]]},
                  {"role": "user", "text": "thanks"}]
    gemini = providers._to_openai_messages(transcript, "gemini")
    assert [m["role"] for m in gemini] == ["system", "user", "assistant", "tool", "assistant", "user"]
    replayed = gemini[2]["tool_calls"][0]
    assert len(replayed["id"]) == 40 and gemini[3]["tool_call_id"] == replayed["id"]
    assert replayed["extra_content"]["google"]["thought_signature"] == providers.DUMMY_THOUGHT_SIGNATURE
    ollama = providers._to_openai_messages(transcript, "ollama")
    assert "extra_content" not in ollama[2]["tool_calls"][0]  # thought signatures are Gemini-only
    claude = providers._to_claude_messages(transcript)
    assert [m["role"] for m in claude] == ["user", "assistant", "user", "assistant", "user"]
    assert claude[1]["content"][0]["id"] == claude[2]["content"][0]["tool_use_id"] == step["id"]


def test_unknown_model():
    with pytest.raises(ProviderError) as e:
        run_turn("nope", [])
    assert e.value.status == 400


def test_unconfigured_providers(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setitem(providers.PROVIDERS["bedrock"], "configured", lambda: False)
    with pytest.raises(ProviderError) as e:
        run_turn("gemini-3.8-flash", [{"role": "user", "text": "x"}])
    assert e.value.status == 503 and "GEMINI_API_KEY" in str(e.value)
    with pytest.raises(ProviderError) as e:
        run_turn("claude-sonnet-5", [{"role": "user", "text": "x"}])
    assert e.value.status == 503 and "AWS_ACCESS_KEY_ID" in str(e.value)


@pytest.mark.parametrize("exc, status, text", [
    (lambda: _status_error(openai.RateLimitError, 429), 429, "free-tier limit"),
    (lambda: openai.RateLimitError("Quota exceeded for metric: ...free_tier_requests, limit: 20", body={"details": [
        {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]},
        response=httpx.Response(429, request=httpx.Request("POST", "https://example.test"))), 429, "free daily limit"),
    (lambda: _status_error(openai.AuthenticationError, 401), 502, "rejected the API key"),
    (lambda: _status_error(openai.NotFoundError, 404), 502, "does not offer"),
    (lambda: _status_error(openai.InternalServerError, 500), 502, "Gemini error"),
    (lambda: openai.APIConnectionError(request=httpx.Request("POST", "http://x")), 503, "Could not reach Gemini"),
])
def test_gemini_errors_map_to_provider_errors(fake_client, exc, status, text):
    fake_client([exc()])
    with pytest.raises(ProviderError) as e:
        run_turn("gemini-3.8-flash", [{"role": "user", "text": "x"}])
    assert e.value.status == status and text in str(e.value)


def test_ollama_models_are_discovered(ollama_models):
    assert not any(m["provider"] == "ollama" for m in providers.list_models())
    ollama_models("qwen2.5:3b")
    listed = [m for m in providers.list_models() if m["provider"] == "ollama"]
    assert listed == [{"id": "ollama/qwen2.5:3b", "label": "qwen2.5:3b", "provider": "ollama",
                       "note": "Free, local (Ollama), 3.1B", "available": True}]


def test_ollama_model_pulled_after_listing_is_found(ollama_models):
    providers.list_models()
    ollama_models("mistral:latest")
    assert providers.get_model("ollama/mistral:latest")["api_model"] == "mistral:latest"
    assert providers.get_model("ollama/missing") is None


def test_ollama_tool_loop(fake_client, ollama_models):
    ollama_models("qwen2.5:3b")
    call = tool_call("get_policy", json.dumps({"topic": "returns"}))
    client = fake_client([completion(tool_calls=[call]), completion("7 days.")])
    reply, steps = run_turn("ollama/qwen2.5:3b", [{"role": "user", "text": "returns?"}])
    assert reply == "7 days." and steps[0][0]["name"] == "get_policy"
    assert client.requests[0]["model"] == "qwen2.5:3b"


@pytest.mark.parametrize("exc, status, text", [
    (lambda: _status_error(openai.NotFoundError, 404), 502, "ollama pull qwen2.5:3b"),
    (lambda: openai.APITimeoutError(request=httpx.Request("POST", "http://x")), 504, "took too long"),
    (lambda: openai.APIConnectionError(request=httpx.Request("POST", "http://x")), 503, "ollama serve"),
])
def test_ollama_errors_map_to_provider_errors(fake_client, ollama_models, exc, status, text):
    ollama_models("qwen2.5:3b")
    fake_client([exc()])
    with pytest.raises(ProviderError) as e:
        run_turn("ollama/qwen2.5:3b", [{"role": "user", "text": "x"}])
    assert e.value.status == status and text in str(e.value)


@pytest.mark.parametrize("exc, status, text", [
    (lambda: _status_error(anthropic.RateLimitError, 429), 429, "rate limiting"),
    (lambda: _status_error(anthropic.InternalServerError, 500), 502, "Bedrock error"),
    (lambda: anthropic.APIConnectionError(request=httpx.Request("POST", "http://x")), 503, "Could not reach"),
])
def test_claude_errors_map_to_provider_errors(fake_bedrock, exc, status, text):
    fake_bedrock([exc()])
    with pytest.raises(ProviderError) as e:
        run_turn("claude-sonnet-5", [{"role": "user", "text": "x"}])
    assert e.value.status == status and text in str(e.value)
