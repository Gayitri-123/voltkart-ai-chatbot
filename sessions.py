"""Where conversations live between requests.

With REDIS_URL set, conversations are kept in Redis, so they survive restarts and deploys and are shared by every
server process. Without it they are kept in this process's memory (local development and tests only).

Both stores behave the same: a conversation is its owner id plus a list of turns (format documented in providers.py),
expires SESSION_TTL_SECONDS after its last turn, and is locked while a reply is being generated so overlapping requests
can't interleave turns. A lock expires after SESSION_LOCK_SECONDS in case the server dies mid-reply; a turn that
outlives its lock, or whose conversation was reset meanwhile, is not recorded.
"""

import json
import os
import secrets
import threading
import time
import uuid

SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", 24 * 3600))
# Longer than the slowest reply: up to MAX_TOOL_ROUNDS model calls, local models taking minutes each.
LOCK_TTL = int(os.environ.get("SESSION_LOCK_SECONDS", 600))

try:
    import redis
    ERRORS = (redis.RedisError,)  # the store couldn't be reached
except ImportError:  # only needed when REDIS_URL is set
    redis = None
    ERRORS = ()


def _copy(turns):
    """A JSON round trip, so the memory store hands out and keeps the same plain data Redis would."""
    return json.loads(json.dumps(turns))


class MemoryStore:
    """Lost on restart and not shared between server processes."""

    def __init__(self, ttl=SESSION_TTL, lock_ttl=LOCK_TTL, clock=time.monotonic):
        self.ttl, self.lock_ttl, self.clock = ttl, lock_ttl, clock
        self._mutex = threading.Lock()
        self._sessions = {}  # session id -> {"owner", "turns", "expires"}
        self._locks = {}  # session id -> (token, expires)

    def _live(self, session_id):
        session = self._sessions.get(session_id)
        if session and session["expires"] <= self.clock():
            del self._sessions[session_id]
            return None
        return session

    def _held(self, session_id, token):
        held = self._locks.get(session_id)
        return held is not None and held[0] == token and held[1] > self.clock()

    def create(self, owner):
        session_id = str(uuid.uuid4())
        with self._mutex:
            for old in [k for k, s in self._sessions.items() if s["expires"] <= self.clock()]:
                del self._sessions[old]
            self._sessions[session_id] = {"owner": owner, "turns": [], "expires": self.clock() + self.ttl}
        return session_id

    def owner(self, session_id):
        with self._mutex:
            session = self._live(session_id)
            return session["owner"] if session else None

    def transcript(self, session_id):
        with self._mutex:
            session = self._live(session_id)
            return _copy(session["turns"]) if session else []

    def lock(self, session_id):
        """A token for the conversation's lock, or None if another reply holds it."""
        with self._mutex:
            held = self._locks.get(session_id)
            if held and held[1] > self.clock():
                return None
            token = secrets.token_hex(16)
            self._locks[session_id] = (token, self.clock() + self.lock_ttl)
            return token

    def unlock(self, session_id, token):
        with self._mutex:
            if self._locks.get(session_id, (None,))[0] == token:
                del self._locks[session_id]

    def append(self, session_id, token, turns):
        """Record turns and restart the expiry, if the lock is still held and the conversation still exists."""
        with self._mutex:
            session = self._live(session_id)
            if session is None or not self._held(session_id, token):
                return False
            session["turns"].extend(_copy(turns))
            session["expires"] = self.clock() + self.ttl
            return True

    def delete(self, session_id):
        with self._mutex:
            self._sessions.pop(session_id, None)


class RedisStore:
    """Keys per conversation: <prefix><id>:owner (string), :turns (list of JSON turns), :lock (token)."""

    def __init__(self, client, ttl=SESSION_TTL, lock_ttl=LOCK_TTL, prefix="chat:"):
        self.r, self.ttl, self.lock_ttl, self.prefix = client, ttl, lock_ttl, prefix

    def _keys(self, session_id):
        base = f"{self.prefix}{session_id}"
        return f"{base}:owner", f"{base}:turns", f"{base}:lock"

    def create(self, owner):
        session_id = str(uuid.uuid4())
        self.r.set(self._keys(session_id)[0], owner, ex=self.ttl)
        return session_id

    def owner(self, session_id):
        return self.r.get(self._keys(session_id)[0])

    def transcript(self, session_id):
        return [json.loads(t) for t in self.r.lrange(self._keys(session_id)[1], 0, -1)]

    def lock(self, session_id):
        token = secrets.token_hex(16)
        return token if self.r.set(self._keys(session_id)[2], token, nx=True, ex=self.lock_ttl) else None

    def unlock(self, session_id, token):
        lock_key = self._keys(session_id)[2]
        self._while_locked(session_id, token, lambda pipe: pipe.delete(lock_key), need_session=False)

    def append(self, session_id, token, turns):
        owner_key, turns_key, _ = self._keys(session_id)

        def write(pipe):
            pipe.rpush(turns_key, *(json.dumps(t) for t in turns))
            pipe.expire(owner_key, self.ttl)
            pipe.expire(turns_key, self.ttl)
        return self._while_locked(session_id, token, write)

    def _while_locked(self, session_id, token, write, need_session=True):
        """Run write in a transaction if the lock still holds token (and the conversation exists). Returns whether it ran."""
        owner_key, _, lock_key = self._keys(session_id)
        with self.r.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(lock_key, owner_key)
                    if pipe.get(lock_key) != token or (need_session and not pipe.exists(owner_key)):
                        pipe.unwatch()
                        return False
                    pipe.multi()
                    write(pipe)
                    pipe.execute()
                    return True
                except redis.WatchError:  # the lock or conversation changed meanwhile; check again
                    continue

    def delete(self, session_id):
        owner_key, turns_key, _ = self._keys(session_id)
        self.r.delete(owner_key, turns_key)


def store_from_env():
    url = os.environ.get("REDIS_URL")
    if not url:
        return MemoryStore()
    if redis is None:
        raise RuntimeError("REDIS_URL is set but the redis package isn't installed (pip install redis).")
    return RedisStore(redis.Redis.from_url(url, decode_responses=True, socket_timeout=5, socket_connect_timeout=5))
