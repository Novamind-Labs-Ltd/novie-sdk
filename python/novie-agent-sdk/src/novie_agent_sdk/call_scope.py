"""Agent 侧拆包 AgentCallScope 的 helper。

Platform 在 invoke payload 里塞 ``inputs["__call_scope__"]``（进 agent
容器时 HttpJsonExternalAgentClient 会把它提成顶层 ``payload["call_scope"]``）。
Agent 侧两条路径都可能看到：

- 进程内（少见）：从 ``inputs["__call_scope__"]`` 拿
- HTTP invoke（常见）：从 ``payload["call_scope"]`` 拿

本 helper 两条都支持。失败时返回 None，agent 走默认行为（shared workspace，
不做 cleanup），不阻塞任务。
"""
from __future__ import annotations

import logging
from typing import Any

from novie_protocol.contracts import AgentCallScope

__all__ = ["extract_call_scope"]

_log = logging.getLogger(__name__)


def extract_call_scope(source: dict[str, Any] | None) -> AgentCallScope | None:
    """从 inputs dict **或** 顶层 payload 里拆出 AgentCallScope。

    容忍两种输入：
    - ``extract_call_scope(inputs)`` — agent 代码拿到的 ``inputs`` 形参
    - ``extract_call_scope(payload)`` — FastAPI / http server 收到的顶层 dict

    返回 None 时 agent 应按保守默认行为跑（即 ``sandbox_isolation=shared`` /
    无 cleanup），而不是 raise。
    """
    if not source:
        return None

    # 优先顶层（http invoke 路径）；其次双下划线（in-process 路径）
    raw = source.get("call_scope")
    if raw is None:
        raw = source.get("__call_scope__")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _log.warning(
            "call_scope present but not a dict (got %s); ignoring",
            type(raw).__name__,
        )
        return None
    try:
        return AgentCallScope.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        _log.warning("failed to parse call_scope: %s; ignoring", exc)
        return None
