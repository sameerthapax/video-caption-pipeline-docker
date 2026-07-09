from __future__ import annotations

import asyncio
from typing import Final
from weakref import WeakKeyDictionary


_SEMAPHORES: Final[WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Semaphore]]] = WeakKeyDictionary()


def get_loop_semaphore(*, name: str, limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    loop_semaphores = _SEMAPHORES.setdefault(loop, {})
    semaphore = loop_semaphores.get(name)
    if semaphore is None:
        semaphore = asyncio.Semaphore(max(1, limit))
        loop_semaphores[name] = semaphore
    return semaphore
