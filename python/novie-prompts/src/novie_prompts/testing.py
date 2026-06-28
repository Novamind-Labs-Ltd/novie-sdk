"""Public CI test double — the single supported seam for all consumer repos (no Langfuse in CI)."""
from __future__ import annotations

from typing import Any

from . import client, telemetry


class RecordingRecorder:
    def __init__(self) -> None:
        self.fallbacks: list[tuple[str, str]] = []
        self.lives: list[str] = []

    def record_fallback(self, name: str, reason: str) -> None:
        self.fallbacks.append((name, reason))

    def record_live(self, name: str) -> None:
        self.lives.append(name)


class _FakePrompt:
    def __init__(self, prompt: Any) -> None:
        self.prompt = prompt


class FakeClient:
    """Stand-in for the Langfuse client. `text` is returned as `.prompt`; `raises` is raised."""

    def __init__(self, *, text: Any = None, raises: BaseException | None = None) -> None:
        self._text = text
        self._raises = raises
        self.last_call: dict[str, Any] | None = None

    def get_prompt(self, name: str, **kwargs: Any) -> _FakePrompt:
        self.last_call = {"name": name, **kwargs}
        if self._raises is not None:
            raise self._raises
        return _FakePrompt(self._text)


def install_fake(
    *, text: Any = None, raises: BaseException | None = None
) -> tuple[FakeClient, RecordingRecorder]:
    fake = FakeClient(text=text, raises=raises)
    rec = RecordingRecorder()
    client.set_client_for_test(fake)
    telemetry.set_recorder(rec)
    return fake, rec


def reset() -> None:
    client.reset_client()
    telemetry.set_recorder(None)


# Spec §3/§8 name this seam `fake_registry`; preserve that published name for the 6 consumer repos.
fake_registry = install_fake
