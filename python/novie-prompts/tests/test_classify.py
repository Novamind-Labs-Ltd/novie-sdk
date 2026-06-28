import httpx
from langfuse.api import NotFoundError
from novie_prompts.registry import _classify


def test_timeout():
    assert _classify(httpx.ReadTimeout("slow")) == "timeout"
    assert _classify(httpx.ConnectTimeout("slow")) == "timeout"


def test_missing_404():
    err = NotFoundError(body="nope")
    assert _classify(err) == "missing"


def test_other():
    assert _classify(ValueError("x")) == "exception"
