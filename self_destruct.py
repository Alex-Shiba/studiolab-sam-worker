"""One-shot idle retirement, independent of the encoder and testable offline."""

import asyncio
import os
import time
from urllib.parse import quote


async def delete_instance(instance_id, key):
    import aiohttp

    url = f"https://console.vast.ai/api/v0/instances/{quote(instance_id, safe='')}/"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as http:
            async with http.delete(url, headers={"Authorization": f"Bearer {key}"},
                                   allow_redirects=False) as response:
                return response.status
    except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
        # Exception messages can contain credentials or URLs; discard them.
        raise OSError("network") from None


class IdleRetirement:
    def __init__(self, log, *, env=None, clock=time.monotonic,
                 delete=delete_instance, sleep=asyncio.sleep):
        env = os.environ if env is None else env
        self.threshold = float(env.get("SAM_SELF_DESTRUCT_IDLE_S", "540"))
        self.instance_id = env.get("CONTAINER_ID")
        self.key = env.get("CONTAINER_API_KEY")
        self.clock, self.delete, self.sleep, self.log = clock, delete, sleep, log
        self.last_activity = clock()
        self.in_flight = 0
        self.done = False

    def request_started(self):
        self.in_flight += 1
        self.last_activity = self.clock()

    def request_finished(self):
        self.in_flight -= 1
        self.last_activity = self.clock()

    async def check(self):
        if self.done:
            return
        reason = None
        if self.threshold == 0:
            reason = "disabled"
        elif not self.key:
            reason = "no_container_key"
        elif not self.instance_id:
            reason = "no_container_id"
        if reason:
            self.done = True
            self.log(f"SAM_SELF_DESTRUCT skipped={reason}")
            return
        idle = self.clock() - self.last_activity
        if self.in_flight or idle < self.threshold:
            return
        # Commit once, before yielding: later activity cannot undo retirement.
        self.done = True
        for attempt in range(4):  # Initial call plus at most three retries.
            try:
                status = await self.delete(self.instance_id, self.key)
            except (OSError, asyncio.TimeoutError):
                status = "network_error"
            self.log(f"SAM_SELF_DESTRUCT idle={int(idle)} status={status}")
            if status != "network_error" and not 500 <= status <= 599:
                return
            if attempt < 3:
                await self.sleep(30)

    async def run(self):
        while not self.done:
            await self.check()
            if not self.done:
                await self.sleep(15)
