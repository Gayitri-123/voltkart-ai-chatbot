import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evals"))


def tool_call(name, arguments, call_id="call_1", signature=None):
    """A fake OpenAI tool call object, shaped like the SDK's."""
    extra = {"extra_content": {"google": {"thought_signature": signature}}} if signature else {}
    return SimpleNamespace(id=call_id, type="function", model_extra=extra,
                           function=SimpleNamespace(name=name, arguments=arguments))


def completion(content=None, tool_calls=None, prompt_tokens=10, completion_tokens=5, cached=0):
    """A fake chat.completions.create response."""
    def model_dump(exclude_none=True):
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                 for c in tool_calls]
        return {k: v for k, v in msg.items() if v is not None}

    message = SimpleNamespace(content=content, tool_calls=tool_calls, model_dump=model_dump)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeClient:
    """Returns queued responses (or raises queued exceptions) and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture(autouse=True)
def no_local_ollama(monkeypatch):
    """Tests never reach a real Ollama server; ollama_models([...]) fakes the pulled models instead."""
    import providers

    monkeypatch.setattr(providers, "_ollama", {})
    found = []
    monkeypatch.setattr(providers, "_ollama_models", lambda: list(found))
    return found


@pytest.fixture
def ollama_models(no_local_ollama):
    def install(*names):
        no_local_ollama[:] = [{"name": n, "digest": n, "details": {"parameter_size": "3.1B"}} for n in names]
    return install


@pytest.fixture
def fake_client(monkeypatch):
    """fake_client([responses...]) installs a FakeClient for Gemini and configures its key."""
    import providers

    monkeypatch.setenv("GEMINI_API_KEY", "test")

    def install(responses):
        client = FakeClient(responses)
        monkeypatch.setattr(providers, "_client", lambda provider: client)
        return client

    return install


def claude_message(text=None, tool_uses=(), stop_reason=None, input_tokens=10, output_tokens=5):
    """A fake Anthropic messages.create response. tool_uses: [(id, name, input), ...]."""
    content = [SimpleNamespace(type="tool_use", id=i, name=n, input=a) for i, n, a in tool_uses]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                            cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=content, usage=usage,
                           stop_reason=stop_reason or ("tool_use" if tool_uses else "end_turn"))


class FakeBedrock:
    """Returns queued Anthropic responses (or raises queued exceptions) and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def fake_bedrock(monkeypatch):
    """fake_bedrock([responses...]) installs a FakeBedrock client and pretends AWS credentials are set."""
    import providers

    monkeypatch.setitem(providers.PROVIDERS["bedrock"], "configured", lambda: True)

    def install(responses):
        client = FakeBedrock(responses)
        monkeypatch.setattr(providers, "_bedrock", lambda: client)
        return client

    return install
