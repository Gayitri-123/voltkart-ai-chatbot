import pytest
from fastapi.testclient import TestClient

import app as app_module
from providers import ProviderError


@pytest.fixture
def client(monkeypatch):
    app_module.sessions.clear()
    return TestClient(app_module.app)


def fake_turn(reply="Hi!", steps=None):
    def run_turn(model_id, transcript, usage):
        usage["input_tokens"] = 3
        fake_turn.seen = list(transcript)
        return reply, steps or []
    return run_turn


def test_index_and_static(client):
    assert "<html" in client.get("/").text.lower()
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_models_endpoint(client):
    data = client.get("/api/models").json()
    assert data["default"] in [m["id"] for m in data["models"]]
    assert data["store_name"]


def test_chat_keeps_session_history(client, monkeypatch):
    monkeypatch.setattr(app_module, "run_turn", fake_turn())
    first = client.post("/api/chat", json={"message": "hello"}).json()
    assert first["reply"] == "Hi!" and first["usage"] == {"input_tokens": 3}
    assert "trace" not in first
    sid = first["session_id"]
    client.post("/api/chat", json={"message": "again", "session_id": sid})
    assert [t["text"] for t in fake_turn.seen] == ["hello", "Hi!", "again"]


def test_chat_debug_trace(client, monkeypatch):
    step = {"id": "1", "name": "get_policy", "input": {"topic": "returns"}, "output": {}}
    monkeypatch.setattr(app_module, "run_turn", fake_turn(steps=[[step]]))
    data = client.post("/api/chat", json={"message": "x", "debug": True}).json()
    assert data["trace"] == {"tools": [step]}


def test_chat_route(client, monkeypatch):
    def step(name):
        return {"id": "1", "name": name, "input": {}, "output": {}}
    monkeypatch.setattr(app_module, "run_turn", fake_turn())
    first = client.post("/api/chat", json={"message": "hi"}).json()
    assert first["route"] == "chat"
    monkeypatch.setattr(app_module, "run_turn", fake_turn(steps=[[step("get_order_status")], [step("get_policy")]]))
    data = client.post("/api/chat", json={"message": "x", "session_id": first["session_id"]}).json()
    assert data["route"] == "order"  # the first lookup; the policy lookup only supports it
    monkeypatch.setattr(app_module, "run_turn", fake_turn())
    data = client.post("/api/chat", json={"message": "and?", "session_id": first["session_id"]}).json()
    assert data["route"] == "order"  # a follow-up without a lookup stays on the earlier route
    monkeypatch.setattr(app_module, "run_turn", fake_turn(steps=[[step("get_policy")]]))
    assert client.post("/api/chat", json={"message": "x"}).json()["route"] == "policy"


def test_chat_unknown_model(client):
    assert client.post("/api/chat", json={"message": "x", "model": "nope"}).status_code == 400


def test_failed_turn_is_not_recorded(client, monkeypatch):
    def boom(*_):
        raise ProviderError(429, "limit")
    monkeypatch.setattr(app_module, "run_turn", boom)
    r = client.post("/api/chat", json={"message": "x", "session_id": "s1"})
    assert r.status_code == 429 and r.json()["detail"] == "limit"
    assert app_module.sessions["s1"] == []


def test_reset(client, monkeypatch):
    monkeypatch.setattr(app_module, "run_turn", fake_turn())
    sid = client.post("/api/chat", json={"message": "x"}).json()["session_id"]
    assert client.post("/api/reset", json={"message": "", "session_id": sid}).json() == {"ok": True}
    assert sid not in app_module.sessions
