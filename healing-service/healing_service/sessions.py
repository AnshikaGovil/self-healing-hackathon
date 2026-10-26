"""Memory per demo session: which stores a request reads and writes.

One public service is shared by everyone who runs the demonstration, and the
demonstration is a story about memory - learned on one release, healed on
the next. One jury's seed must not be another's warm start, so a caller may
name a session (the ``X-BIS-Session`` header) and get baseline and repair
stores of its own, under ``<baseline dir>/sessions/<id>/``. No session, or a
header that is not a plain id, is the shared memory, exactly as before.

The session travels in a ``ContextVar`` set once per request by
``SessionMiddleware``, so every store access on that request - the
endpoints, the ReFlex Agent's tools - lands in the same memory without each
call site being told.
"""

from __future__ import annotations

import contextvars
import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Optional, Tuple

from .baselines import BaselineStore
from .repairs import RepairStore

log = logging.getLogger("bis.service")

HEADER = b"x-bis-session"
# The id becomes a folder name, so nothing that could climb out of one.
_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

current: contextvars.ContextVar = contextvars.ContextVar("bis_session",
                                                         default="")


def session_id(raw: str) -> str:
    """The session a header names, or "" for the shared memory."""
    raw = (raw or "").strip()
    return raw if _ID.match(raw) else ""


class SessionStores:
    """Each session's baseline and repair stores, opened when first asked.

    At most ``limit`` are open at once; the least recently used is saved and
    closed (a closed repair store degrades to no-ops, so a request still
    holding one fails open). At most ``most`` sessions exist on disk: the
    header naming one is anyone's to send, and past the cap a new id is
    refused - ``get`` returns None and the caller uses the shared memory.
    """

    def __init__(self, folder: str, limit: int = 64, most: int = 500):
        self.folder = folder
        self.limit = max(1, limit)
        self.most = max(0, most)
        self._lock = threading.Lock()
        self._open: "OrderedDict[str, Tuple[BaselineStore, RepairStore]]" = (
            OrderedDict())
        try:
            self._known = {name for name in os.listdir(folder)
                           if os.path.isdir(os.path.join(folder, name))}
        except OSError:
            self._known = set()

    def get(self, sid: str) -> Optional[Tuple[BaselineStore, RepairStore]]:
        """``(baselines, repairs)`` for this session, or None past the cap."""
        with self._lock:
            pair = self._open.get(sid)
            if pair is not None:
                self._open.move_to_end(sid)
                return pair
            if sid not in self._known and len(self._known) >= self.most:
                log.warning("session %s refused - %d sessions already exist "
                            "(BIS_MAX_SESSIONS); it shares the common memory",
                            sid, len(self._known))
                return None
            where = os.path.join(self.folder, sid)
            pair = (BaselineStore(os.path.join(where, "baselines.json")),
                    RepairStore(os.path.join(where, "repairs.db")))
            self._known.add(sid)
            self._open[sid] = pair
            while len(self._open) > self.limit:
                _gone, (baselines, repairs) = self._open.popitem(last=False)
                baselines.save()
                repairs.close()
            return pair

    def close(self) -> None:
        """Save and close every open session's stores."""
        with self._lock:
            for baselines, repairs in self._open.values():
                baselines.save()
                repairs.close()
            self._open.clear()


class SessionMiddleware:
    """Names the session each HTTP request's memory belongs to.

    Plain ASGI rather than BaseHTTPMiddleware: the value is set in the task
    that runs the endpoint, and a sync endpoint's worker thread starts from
    a copy of that task's context.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        raw = ""
        for name, value in scope.get("headers") or []:
            if name == HEADER:
                raw = value.decode("latin-1")
                break
        token = current.set(session_id(raw))
        try:
            await self.app(scope, receive, send)
        finally:
            current.reset(token)
