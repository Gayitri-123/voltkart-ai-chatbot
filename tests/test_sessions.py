import threading

import pytest

import sessions

TURNS = [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "Hello!", "steps": [[{"name": "get_policy",
          "input": {"topic": "returns"}, "output": {"price": 19999, "in_stock": True, "mrp": 1.5}}]]}]


@pytest.fixture(params=["memory", "redis"])
def store(request):
    if request.param == "memory":
        return sessions.MemoryStore()
    fakeredis = pytest.importorskip("fakeredis")
    return sessions.RedisStore(fakeredis.FakeRedis(decode_responses=True))


def test_round_trip(store):
    sid = store.create("owner-1")
    assert store.owner(sid) == "owner-1" and store.transcript(sid) == []
    token = store.lock(sid)
    assert store.append(sid, token, TURNS) and store.append(sid, token, TURNS[:1])
    assert store.transcript(sid) == TURNS + TURNS[:1]
    store.unlock(sid, token)


def test_unknown_session(store):
    assert store.owner("nope") is None and store.transcript("nope") == []
    token = store.lock("nope")
    assert not store.append("nope", token, TURNS)  # never creates a conversation
    assert store.owner("nope") is None


def test_lock_is_exclusive_per_conversation(store):
    a, b = store.create("o"), store.create("o")
    token = store.lock(a)
    assert token and store.lock(a) is None and store.lock(b)
    assert not store.append(a, "someone-else", TURNS)
    store.unlock(a, "someone-else")  # a wrong token doesn't release it
    assert store.lock(a) is None
    store.unlock(a, token)
    assert store.lock(a)


def test_only_one_of_many_concurrent_requests_gets_the_lock(store):
    sid = store.create("o")
    barrier = threading.Barrier(8)
    tokens = []

    def take():
        barrier.wait()
        tokens.append(store.lock(sid))
    threads = [threading.Thread(target=take) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len([t for t in tokens if t]) == 1


def test_delete(store):
    sid = store.create("o")
    token = store.lock(sid)
    store.append(sid, token, TURNS)
    store.delete(sid)
    assert store.owner(sid) is None and store.transcript(sid) == []
    assert not store.append(sid, token, TURNS)  # a reply finishing after a reset isn't recorded


def test_turns_are_copied(store):
    sid = store.create("o")
    token = store.lock(sid)
    turns = [dict(t) for t in TURNS]
    store.append(sid, token, turns)
    turns[0]["text"] = "changed"
    store.transcript(sid)[0]["text"] = "changed too"
    assert store.transcript(sid) == TURNS


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def test_memory_expiry():
    clock = Clock()
    store = sessions.MemoryStore(ttl=60, lock_ttl=10, clock=clock)
    sid = store.create("o")
    clock.now += 50
    token = store.lock(sid)
    assert store.append(sid, token, TURNS)  # a turn restarts the expiry
    clock.now += 50
    assert store.owner(sid) == "o"
    clock.now += 11
    assert store.lock(sid)  # the lock of a reply that died is taken over...
    assert not store.append(sid, token, TURNS)  # ...and the stale reply can't write any more
    clock.now += 60
    assert store.owner(sid) is None and store.transcript(sid) == []
    store.create("o")  # creating a conversation sweeps expired ones
    assert sid not in store._sessions


def test_redis_keys_expire():
    fakeredis = pytest.importorskip("fakeredis")
    r = fakeredis.FakeRedis(decode_responses=True)
    store = sessions.RedisStore(r, ttl=60, lock_ttl=10)
    sid = store.create("o")
    token = store.lock(sid)
    store.append(sid, token, TURNS)
    owner_key, turns_key, lock_key = store._keys(sid)
    assert 0 < r.ttl(owner_key) <= 60 and 0 < r.ttl(turns_key) <= 60 and 0 < r.ttl(lock_key) <= 10
    r.delete(lock_key)  # the lock expired and another request took it
    assert store.lock(sid) and not store.append(sid, token, TURNS)


def test_redis_conversations_outlive_the_process():
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeServer()
    first = sessions.RedisStore(fakeredis.FakeRedis(server=server, decode_responses=True))
    sid = first.create("o")
    token = first.lock(sid)
    first.append(sid, token, TURNS)
    first.unlock(sid, token)
    # A restarted or second server process sees the same conversation.
    second = sessions.RedisStore(fakeredis.FakeRedis(server=server, decode_responses=True))
    assert second.owner(sid) == "o" and second.transcript(sid) == TURNS


def test_store_from_env(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert isinstance(sessions.store_from_env(), sessions.MemoryStore)
    pytest.importorskip("redis")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    assert isinstance(sessions.store_from_env(), sessions.RedisStore)  # connects lazily
