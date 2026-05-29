"""Canonical platform LLM wire helpers.

The platform accepts provider/LangChain/OpenAI-compatible shapes at the
boundary, but SDK consumers should only see one tool-call representation:

    {"id": str, "name": str, "args": dict}

Stream chunks use the LangChain-compatible delta shape:

    {"id": str, "name": str, "args": str, "index": int}

These helpers deliberately strip provider-native ``function`` wrappers and
drop blank-name tool calls. Replaying blank tool calls into the next model
turn makes providers fail before the agent can recover.
"""
from __future__ import annotations

import json
from typing import Any


def normalise_tool_call_chunks(
    chunks: Any,
    state: dict[Any, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return canonical streamed tool-call chunks with stable string ids."""
    if state is None:
        state = {}
    normalised: list[dict[str, Any]] = []
    if not isinstance(chunks, list):
        return normalised

    ids_by_index: dict[int, str] = state.setdefault("ids_by_index", {})
    id_to_index: dict[str, int] = state.setdefault("id_to_index", {})
    for ordinal, raw_piece in enumerate(chunks):
        if not isinstance(raw_piece, dict):
            continue
        piece = _flatten_provider_tool_shape(raw_piece)
        incoming_id = str(piece.get("id") or "")
        name = str(piece.get("name") or "")
        args_text = _args_to_text(piece.get("args"))

        raw_index = piece.get("index")
        if raw_index is not None:
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                index = ordinal
        elif incoming_id:
            index = id_to_index.get(incoming_id)
            if index is None:
                index = int(state.get("next_index", 0))
                state["next_index"] = index + 1
        elif ordinal in ids_by_index:
            index = ordinal
        elif "last_index" in state:
            index = int(state["last_index"])
        else:
            if not name:
                continue
            index = int(state.get("next_index", ordinal))

        if incoming_id:
            stable_id = incoming_id
            ids_by_index[index] = stable_id
            id_to_index[stable_id] = index
            state["last_index"] = index
            state["next_index"] = max(int(state.get("next_index", 0)), index + 1)
        else:
            stable_id = ids_by_index.get(index)
            if not stable_id:
                if not name:
                    continue
                stable_id = f"call_{index}"
                ids_by_index[index] = stable_id
                id_to_index[stable_id] = index

        normalised.append(
            {
                "id": stable_id,
                "name": name,
                "args": args_text,
                "index": index,
            }
        )
    return normalised


def normalise_tool_calls(calls: Any) -> list[dict[str, Any]]:
    """Return canonical aggregated tool calls with parsed dict args."""
    normalised: list[dict[str, Any]] = []
    if not isinstance(calls, list):
        return normalised
    for index, raw_call in enumerate(calls):
        if not isinstance(raw_call, dict):
            continue
        call = _flatten_provider_tool_shape(raw_call)
        name = str(call.get("name") or "")
        if not name:
            continue
        normalised.append(
            {
                "id": str(call.get("id") or f"call_{index}"),
                "name": name,
                "args": _args_to_dict(call.get("args")),
            }
        )
    return normalised


def sanitize_additional_kwargs(value: Any) -> dict[str, Any]:
    """Drop provider-native tool-call payloads from LangChain metadata.

    LangChain messages may carry both canonical ``message.tool_calls`` and
    provider raw ``additional_kwargs["tool_calls"]``. Keeping the latter lets
    invalid provider fragments bypass the canonical contract on replay.
    """
    if not isinstance(value, dict):
        return {}
    sanitized = dict(value)
    sanitized.pop("tool_calls", None)
    sanitized.pop("function_call", None)
    return sanitized


def normalise_llm_result(value: Any) -> dict[str, Any]:
    """Return a platform LLM result with canonical tool-call payloads."""
    if not isinstance(value, dict):
        return {}
    result = dict(value)
    additional_kwargs = result.get("additional_kwargs")
    raw_tool_calls = result.get("tool_calls") or []
    if not raw_tool_calls and isinstance(additional_kwargs, dict):
        raw_tool_calls = additional_kwargs.get("tool_calls") or []
    tool_calls = normalise_tool_calls(raw_tool_calls)
    if tool_calls:
        result["tool_calls"] = tool_calls
    else:
        result.pop("tool_calls", None)
    if "tool_call_chunks" in result:
        result["tool_call_chunks"] = normalise_tool_call_chunks(
            result.get("tool_call_chunks") or [],
            {},
        )
    if additional_kwargs:
        sanitized = sanitize_additional_kwargs(additional_kwargs)
        if sanitized:
            result["additional_kwargs"] = sanitized
        else:
            result.pop("additional_kwargs", None)
    return result


class ToolCallAccumulator:
    """Accumulate canonical stream chunks into complete tool calls.

    Non-LangChain agents can use this with ``ctx.platform.llm.stream_chat`` to
    avoid hand-rolling provider-specific chunk stitching.
    """

    def __init__(self) -> None:
        self._normalise_state: dict[Any, Any] = {}
        self._order: list[str] = []
        self._names: dict[str, str] = {}
        self._arg_parts: dict[str, list[str]] = {}

    def add_chunks(self, chunks: Any) -> list[dict[str, Any]]:
        normalised = normalise_tool_call_chunks(chunks, self._normalise_state)
        for chunk in normalised:
            tool_call_id = str(chunk.get("id") or "")
            if not tool_call_id:
                continue
            if tool_call_id not in self._order:
                self._order.append(tool_call_id)
            name = str(chunk.get("name") or "")
            if name:
                self._names[tool_call_id] = name
            args = chunk.get("args")
            if isinstance(args, str) and args:
                self._arg_parts.setdefault(tool_call_id, []).append(args)
        return normalised

    def add_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        if str(event.get("type") or "") != "chunk":
            return []
        delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
        return self.add_chunks(delta.get("tool_call_chunks") or [])

    def tool_calls(self) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for index, tool_call_id in enumerate(self._order):
            name = self._names.get(tool_call_id, "")
            if not name:
                continue
            raw_args = "".join(self._arg_parts.get(tool_call_id, []))
            calls.append(
                {
                    "id": tool_call_id or f"call_{index}",
                    "name": name,
                    "args": _args_to_dict(raw_args),
                }
            )
        return calls


def normalise_stream_event(
    event: dict[str, Any],
    state: dict[Any, Any] | None = None,
) -> dict[str, Any]:
    """Return an event whose tool-call chunks use the canonical delta shape."""
    event_type = str(event.get("type") or "")
    if event_type == "completed":
        out = dict(event)
        result = out.get("result")
        if isinstance(result, dict):
            out["result"] = normalise_llm_result(result)
        return out
    if event_type != "chunk":
        return dict(event)
    out = dict(event)
    delta = out.get("delta") if isinstance(out.get("delta"), dict) else {}
    normalised_delta = dict(delta)
    normalised_delta["tool_call_chunks"] = normalise_tool_call_chunks(
        normalised_delta.get("tool_call_chunks") or [],
        state,
    )
    out["delta"] = normalised_delta
    return out


def _flatten_provider_tool_shape(value: dict[str, Any]) -> dict[str, Any]:
    out = dict(value)
    function_payload = out.get("function")
    if isinstance(function_payload, dict):
        if out.get("name") is None and function_payload.get("name") is not None:
            out["name"] = function_payload.get("name")
        if out.get("args") is None and function_payload.get("arguments") is not None:
            out["args"] = function_payload.get("arguments")
    if out.get("args") is None and out.get("arguments") is not None:
        out["args"] = out.get("arguments")
    return out


def _args_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _args_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value) if value else {}
        except (json.JSONDecodeError, ValueError):
            parsed = {"_raw": value}
        return parsed if isinstance(parsed, dict) else {"_raw": parsed}
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return {"_raw": value}
