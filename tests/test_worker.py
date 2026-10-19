import logging

from aiohttp import web, ClientSession, TCPConnector, ThreadedResolver
from aiohttp.test_utils import TestClient, TestServer

import worker
from contract import HEADER, WIRE_SIZE
from vastai.serverless.server.worker import EndpointHandlerFactory


async def test_sdk_binary_passthrough():
    # Exercise the installed SDK's default response path without routing or keys.
    body = HEADER + bytes(WIRE_SIZE - 8)

    async def embed(request):
        return web.Response(body=body, content_type="application/octet-stream")

    model = web.Application()
    model.router.add_post("/embed", embed)
    handler = EndpointHandlerFactory(worker.build_config()).get_benchmark_handler()
    assert handler.endpoint == "/embed"
    async with TestServer(model) as backend, ClientSession(connector=TCPConnector(resolver=ThreadedResolver())) as session:
        async def proxy(request):
            async with session.post(backend.make_url("/embed"), json={}) as response:
                return await handler.generate_client_response(request, response)

        app = web.Application()
        app.router.add_post("/embed", proxy)
        async with TestClient(TestServer(app), connector=TCPConnector(resolver=ThreadedResolver())) as http:
            response = await http.post("/embed")
            assert response.status == 200
            assert response.headers["Content-Type"] == "application/octet-stream"
            assert await response.read() == body


def test_worker_owns_child_and_disables_payload_debug(tmp_path, monkeypatch):
    events = []

    class Child:
        def __init__(self, argv):
            events.append("start")
            assert argv[-1].endswith("server.py")

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            events.append("wait")

    class Proxy:
        def __init__(self, config):
            logging.getLogger().setLevel(logging.DEBUG)

        def run(self):
            assert not logging.getLogger("vast-payload-test").isEnabledFor(logging.DEBUG)
            events.append("run")

    monkeypatch.setenv("SAM_LOG_FILE", str(tmp_path / "sam.log"))
    (tmp_path / "sam.log").write_text("SAM_READY\n")
    monkeypatch.setattr(worker.subprocess, "Popen", Child)
    monkeypatch.setattr(worker, "Worker", Proxy)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    old_level = logging.getLogger().level
    old_disable = logging.root.manager.disable
    try:
        worker.main()
        assert events == ["start", "run", "terminate", "wait"]
        assert (tmp_path / "sam.log").read_text() == ""
    finally:
        logging.disable(old_disable)
        logging.getLogger().setLevel(old_level)
