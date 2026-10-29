import threading

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
    monkeypatch.setattr(app_module, "DEBUG_TOKEN", "op-secret")
    step = {"id": "1", "name": "get_policy", "input": {"topic": "returns"}, "output": {}}
    monkeypatch.setattr(app_module, "run_turn", fake_turn(steps=[[step]]))
    data = client.post("/api/chat", json={"message": "x", "debug": True}, headers={"X-Debug-Token": "op-secret"}).json()
    assert data["trace"] == {"tools": [step]}


@pytest.mark.parametrize("token, headers", [
    ("", {}),                                    # debug traces disabled (production default)
    ("", {"X-Debug-Token": ""}),
    ("op-secret", {}),                           # no operator token
    ("op-secret", {"X-Debug-Token": "guess"}),   # wrong operator token
])
def test_chat_debug_trace_needs_operator(client, monkeypatch, token, headers):
    monkeypatch.setattr(app_module, "DEBUG_TOKEN", token)
    step = {"id": "1", "name": "get_policy", "input": {"topic": "returns"}, "output": {}}
    monkeypatch.setattr(app_module, "run_turn", fake_turn(steps=[[step]]))
    r = client.post("/api/chat", json={"message": "x", "debug": True}, headers=headers)
    assert r.status_code == 200 and "trace" not in r.json()


def test_chat_debug_trace_is_redacted(client, monkeypatch):
    monkeypatch.setattr(app_module, "DEBUG_TOKEN", "op-secret")
    # Synthetic personal data, never a real customer.
    step = {"id": "1", "name": "get_order_status", "input": {"order_id": "ORD10002", "phone": "+91 98765 43210"},
            "output": {"order_id": "ORD10002", "status": "Out for delivery",
                       "note": "Call 9876543210 or mail test.user@example.com", "items": [{"price": 19999}]}}
    monkeypatch.setattr(app_module, "run_turn", fake_turn(steps=[[step]]))
    data = client.post("/api/chat", json={"message": "x", "debug": True}, headers={"X-Debug-Token": "op-secret"}).json()
    call = data["trace"]["tools"][0]
    assert call["input"] == {"order_id": "ORD10002", "phone": "***3210"}
    assert call["output"]["note"] == "Call ******3210 or mail ***@***"
    assert call["output"]["items"] == [{"price": 19999}] and call["output"]["status"] == "Out for delivery"
    assert "9876543210" not in str(data) and "example.com" not in str(data)
    # The stored transcript keeps the real values for the model.
    assert app_module.sessions[data["session_id"]].transcript[-1]["steps"] == [[step]]


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
    r = client.post("/api/chat", json={"message": "x"})
    assert r.status_code == 429 and r.json()["detail"] == "limit"
    [session] = app_module.sessions.values()
    assert session.transcript == []
    assert not session.lock.locked()  # a failed turn doesn't leave the conversation stuck


def test_reset(client, monkeypatch):
    monkeypatch.setattr(app_module, "run_turn", fake_turn())
    sid = client.post("/api/chat", json={"message": "x"}).json()["session_id"]
    assert client.post("/api/reset", json={"message": "", "session_id": sid}).json() == {"ok": True}
    assert sid not in app_module.sessions


def test_unknown_session_is_rejected(client, monkeypatch):
    monkeypatch.setattr(app_module, "run_turn", fake_turn())
    r = client.post("/api/chat", json={"message": "x", "session_id": "made-up"})
    assert r.status_code == 404
    assert "made-up" not in app_module.sessions


def test_copied_session_id_is_rejected(monkeypatch):
    app_module.sessions.clear()
    monkeypatch.setattr(app_module, "run_turn", fake_turn())
    alice, mallory = TestClient(app_module.app), TestClient(app_module.app)
    sid = alice.post("/api/chat", json={"message": "my order ORD10001, phone 9876543210"}).json()["session_id"]

    # Another browser with the copied session id can't read, append to or reset the conversation...
    assert mallory.post("/api/chat", json={"message": "what did I just say?", "session_id": sid}).status_code == 404
    assert mallory.post("/api/reset", json={"message": "", "session_id": sid}).status_code == 404
    assert len(app_module.sessions[sid].transcript) == 2
    # ...nor can a caller with no cookie at all.
    no_cookie = TestClient(app_module.app)
    assert no_cookie.post("/api/reset", json={"message": "", "session_id": sid}).status_code == 404

    # The owner still can.
    assert alice.post("/api/chat", json={"message": "again", "session_id": sid}).status_code == 200
    assert [t["text"] for t in fake_turn.seen][-1] == "again"
    assert alice.post("/api/reset", json={"message": "", "session_id": sid}).status_code == 200
    assert sid not in app_module.sessions


def test_owner_cookie_is_http_only(client, monkeypatch):
    monkeypatch.setattr(app_module, "run_turn", fake_turn())
    cookie = client.post("/api/chat", json={"message": "x"}).headers["set-cookie"].lower()
    assert app_module.OWNER_COOKIE in cookie and "httponly" in cookie and "samesite=lax" in cookie


def test_overlapping_requests_in_one_session(client, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def slow_turn(model_id, transcript, usage):
        if transcript[-1]["text"] == "slow":
            entered.set()
            assert release.wait(5)
        return f"re: {transcript[-1]['text']} after {len(transcript) - 1}", []

    monkeypatch.setattr(app_module, "run_turn", slow_turn)
    sid = client.post("/api/chat", json={"message": "first"}).json()["session_id"]
    other = client.post("/api/chat", json={"message": "other"}).json()["session_id"]

    results = {}
    slow = threading.Thread(target=lambda: results.update(
        slow=client.post("/api/chat", json={"message": "slow", "session_id": sid})))
    slow.start()
    assert entered.wait(5)
    try:
        # Same conversation while a reply is still being generated: rejected clearly, nothing recorded.
        r = client.post("/api/chat", json={"message": "overlap", "session_id": sid})
        assert r.status_code == 409 and "previous message" in r.json()["detail"]
        # A different conversation isn't held up by it.
        r = client.post("/api/chat", json={"message": "meanwhile", "session_id": other})
        assert r.status_code == 200 and r.json()["reply"] == "re: meanwhile after 2"
    finally:
        release.set()
        slow.join(5)

    assert results["slow"].status_code == 200
    assert [t["text"] for t in app_module.sessions[sid].transcript] == [
        "first", "re: first after 0", "slow", "re: slow after 2"]
    # Once the reply is in, the conversation takes the next message, which sees every earlier turn.
    assert client.post("/api/chat", json={"message": "next", "session_id": sid}).json()["reply"] == "re: next after 4"
