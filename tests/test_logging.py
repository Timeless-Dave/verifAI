import json
import logging
import sys

from verifai.logging import JsonFormatter, request_id_var


def test_log_line_is_json_and_carries_the_request_id() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="verifai.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="citation check failed",
        args=(),
        exc_info=None,
    )
    token = request_id_var.set("abc123")
    try:
        payload = json.loads(formatter.format(record))
    finally:
        request_id_var.reset(token)
    assert payload == {
        "level": "INFO",
        "logger": "verifai.test",
        "message": "citation check failed",
        "request_id": "abc123",
    }


def _boom() -> None:
    raise RuntimeError("boom")


def test_log_line_includes_a_traceback() -> None:
    formatter = JsonFormatter()
    try:
        _boom()
    except RuntimeError:
        record = logging.LogRecord(
            name="verifai.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )
    payload = json.loads(formatter.format(record))
    assert payload["message"] == "failed"
    assert "RuntimeError: boom" in payload["exception"]
    assert "request_id" not in payload
