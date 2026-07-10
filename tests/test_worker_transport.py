from __future__ import annotations

import os
import time
from multiprocessing.connection import Connection

import pytest

from asr_server.errors import AsrError
from asr_server.workers.transport import ProcessRpcTransport


def transport_test_worker(conn: Connection, mode: str) -> None:
    if mode == "startup_crash":
        conn.close()
        return
    try:
        while True:
            request = conn.recv()
            request_id = request["id"]
            op = request["op"]
            if op == "ping":
                conn.send({"id": request_id, "ok": True, "result": {"pid": os.getpid()}})
            elif mode == "hang":
                time.sleep(60)
            elif mode == "oom":
                conn.send(
                    {
                        "id": request_id,
                        "ok": False,
                        "error": {
                            "status_code": 503,
                            "code": "gpu_unavailable",
                            "message": "CUDA out of memory",
                            "details": {"phase": "inference"},
                        },
                    }
                )
            elif op == "shutdown":
                conn.send({"id": request_id, "ok": True, "result": None})
                return
            else:
                conn.send({"id": request_id, "ok": True, "result": "ok"})
    except EOFError:
        return
    finally:
        conn.close()


def make_transport(mode: str) -> ProcessRpcTransport:
    return ProcessRpcTransport(
        name="test",
        target=transport_test_worker,
        target_args=(mode,),
        startup_timeout_seconds=2,
        terminate_timeout_seconds=0.2,
        kill_timeout_seconds=0.2,
    )


def test_transport_round_trip_and_shutdown() -> None:
    transport = make_transport("ok")
    transport.start()
    assert transport.request("work", timeout_seconds=1) == "ok"
    transport.shutdown(timeout_seconds=1)
    assert transport.is_alive is False


def test_transport_times_out_and_reaps_alive_worker() -> None:
    transport = make_transport("hang")
    transport.start()
    pid = transport.pid

    with pytest.raises(AsrError) as exc_info:
        transport.request("work", timeout_seconds=0.05)

    assert exc_info.value.code == "worker_unavailable"
    assert exc_info.value.details["worker_fatal"] is True
    assert exc_info.value.details["worker_pid"] == pid
    assert transport.is_alive is False


def test_transport_reaps_worker_after_cuda_oom() -> None:
    transport = make_transport("oom")
    transport.start()

    with pytest.raises(AsrError) as exc_info:
        transport.request("work", timeout_seconds=1)

    assert exc_info.value.code == "gpu_unavailable"
    assert exc_info.value.details["worker_fatal"] is True
    assert transport.is_alive is False


def test_transport_detects_startup_crash() -> None:
    transport = make_transport("startup_crash")

    with pytest.raises(AsrError) as exc_info:
        transport.start()

    assert exc_info.value.code == "worker_unavailable"
    assert transport.is_alive is False
