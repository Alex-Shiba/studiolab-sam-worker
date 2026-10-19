import asyncio
import selectors
import unittest
from unittest.mock import AsyncMock, patch

from self_destruct import IdleRetirement, delete_instance


class OfflineSelector(selectors.SelectSelector):
    def select(self, timeout=None):
        return []


class OfflineLoop(asyncio.SelectorEventLoop):
    """These tests have no I/O or threads; avoid Windows loopback socketpairs."""

    def __init__(self):
        super().__init__(OfflineSelector())

    def _make_self_pipe(self):
        pass

    def _close_self_pipe(self):
        pass


class RetirementTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.loop_patch = patch("asyncio.events.new_event_loop", OfflineLoop)
        cls.loop_patch.start()
        cls.addClassCleanup(cls.loop_patch.stop)
        cls.alias_patch = patch("asyncio.new_event_loop", OfflineLoop)
        cls.alias_patch.start()
        cls.addClassCleanup(cls.alias_patch.stop)

    def setUp(self):
        self.now = 100
        self.logs = []
        self.env = {"CONTAINER_ID": "123", "CONTAINER_API_KEY": "private-key"}
        self.delete = AsyncMock(return_value=200)
        self.sleep = AsyncMock()
        self.idle = self.make_idle()

    def make_idle(self):
        return IdleRetirement(self.logs.append, env=self.env, clock=lambda: self.now,
                              delete=self.delete, sleep=self.sleep)

    async def test_threshold_and_one_shot(self):
        self.now = 639
        await self.idle.check()
        self.delete.assert_not_awaited()
        self.now = 640
        await self.idle.check()
        await self.idle.check()
        self.delete.assert_awaited_once_with("123", "private-key")
        self.assertEqual(self.logs, ["SAM_SELF_DESTRUCT idle=540 status=200"])

    async def test_in_flight_and_activity_at_both_edges(self):
        self.now = 200
        self.idle.request_started()
        self.assertEqual(self.idle.last_activity, 200)
        self.idle.request_started()
        self.now = 1000
        await self.idle.check()
        self.idle.request_finished()
        self.now = 1600
        await self.idle.check()
        self.delete.assert_not_awaited()
        self.idle.request_finished()
        self.assertEqual(self.idle.last_activity, 1600)
        self.now = 2139
        await self.idle.check()
        self.delete.assert_not_awaited()
        self.now = 2140
        await self.idle.check()
        self.delete.assert_awaited_once()

    async def test_missing_environment_logs_once(self):
        for missing in ("CONTAINER_ID", "CONTAINER_API_KEY"):
            with self.subTest(missing=missing):
                self.env = {"CONTAINER_ID": "123", "CONTAINER_API_KEY": "private-key"}
                del self.env[missing]
                idle = self.make_idle()
                before = len(self.logs)
                await idle.run()
                await idle.check()
                self.assertEqual(len(self.logs), before + 1)
        self.delete.assert_not_awaited()
        self.sleep.assert_not_awaited()

    async def test_disabled(self):
        self.env["SAM_SELF_DESTRUCT_IDLE_S"] = "0"
        idle = self.make_idle()
        self.now = 100000
        await idle.run()
        self.delete.assert_not_awaited()
        self.assertEqual(self.logs, ["SAM_SELF_DESTRUCT skipped=disabled"])

    async def test_custom_threshold_and_poll_interval(self):
        self.env["SAM_SELF_DESTRUCT_IDLE_S"] = "45"
        idle = self.make_idle()

        async def sleep(seconds):
            self.now += seconds

        self.sleep.side_effect = sleep
        await idle.run()
        self.assertEqual([call.args for call in self.sleep.await_args_list], [(15,)] * 3)
        self.delete.assert_awaited_once()

    async def test_retry_limit_network_and_5xx_and_safe_logs(self):
        self.now = 640
        self.delete.side_effect = [OSError("private-key https://host/?secret SAM_FATAL"),
                                   asyncio.TimeoutError("private-key"), 503, 500]
        await self.idle.check()
        await self.idle.check()
        self.assertEqual(self.delete.await_count, 4)
        self.assertEqual([call.args for call in self.sleep.await_args_list], [(30,)] * 3)
        self.assertEqual(len(self.logs), 4)
        for line in self.logs:
            for secret in ("private-key", "https://", "?", "SAM_FATAL"):
                self.assertNotIn(secret, line)

    async def test_no_retry_on_4xx_or_success(self):
        for status in (200, 204, 400, 401, 403, 404, 429):
            with self.subTest(status=status):
                self.delete.reset_mock(side_effect=True)
                self.delete.return_value = status
                idle = self.make_idle()
                self.now += 540
                await idle.check()
                self.delete.assert_awaited_once()
        self.sleep.assert_not_awaited()

    async def test_activity_after_send_does_not_cancel_retries(self):
        self.now = 640
        self.delete.side_effect = [503, 200]

        async def sleep(seconds):
            self.now += seconds
            self.idle.request_started()

        self.sleep.side_effect = sleep
        await self.idle.check()
        self.assertEqual(self.delete.await_count, 2)
        self.assertEqual(self.idle.last_activity, 670)
        self.idle.request_finished()
        await self.idle.check()
        self.assertEqual(self.delete.await_count, 2)

    async def test_http_contract_without_network(self):
        # Supply a fake aiohttp module so this test needs only the stdlib.
        import types
        response = AsyncMock()
        response.__aenter__.return_value.status = 200
        session = AsyncMock()
        from unittest.mock import Mock
        session.delete = Mock(return_value=response)
        context = AsyncMock()
        context.__aenter__.return_value = session
        aiohttp = types.SimpleNamespace(ClientSession=Mock(return_value=context),
                                        ClientTimeout=Mock(), ClientError=OSError)
        with patch.dict("sys.modules", aiohttp=aiohttp):
            self.assertEqual(await delete_instance("123", "private-key"), 200)
        aiohttp.ClientTimeout.assert_called_once_with(total=20)
        session.delete.assert_called_once_with(
            "https://console.vast.ai/api/v0/instances/123/",
            headers={"Authorization": "Bearer private-key"}, allow_redirects=False)

    async def test_server_middleware_and_task_lifecycle(self):
        # Load the actual app factory without importing GPU or HTTP dependencies.
        import ast
        from contextlib import suppress
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import Mock

        source = Path(__file__).resolve().parents[1] / "server.py"
        factory = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                       if isinstance(node, ast.FunctionDef) and node.name == "make_app")
        app = SimpleNamespace(cleanup_ctx=[], router=Mock())
        web = SimpleNamespace(Application=Mock(return_value=app), middleware=lambda fn: fn)
        namespace = {"web": web, "asyncio": asyncio, "suppress": suppress}
        exec(compile(ast.Module(body=[factory], type_ignores=[]), str(source), "exec"), namespace)
        namespace["make_app"](None, self.idle)
        middleware = web.Application.call_args.kwargs["middlewares"][0]

        async def handler(request):
            self.assertEqual(self.idle.in_flight, 1)
            self.assertEqual(self.idle.last_activity, self.now)
            self.now += 10
            raise ValueError("bad request")

        self.now = 200
        with self.assertRaises(ValueError):
            await middleware(SimpleNamespace(method="POST", path="/embed"), handler)
        self.assertEqual(self.idle.in_flight, 0)
        self.assertEqual(self.idle.last_activity, 210)
        self.now = 300
        await middleware(SimpleNamespace(method="GET", path="/health"), AsyncMock())
        self.assertEqual(self.idle.last_activity, 210)

        started = asyncio.Event()
        stopped = asyncio.Event()

        async def run():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        self.idle.run = run
        context = app.cleanup_ctx[0](app)
        await anext(context)
        await started.wait()
        with self.assertRaises(StopAsyncIteration):
            await anext(context)
        self.assertTrue(stopped.is_set())


if __name__ == "__main__":
    unittest.main()
