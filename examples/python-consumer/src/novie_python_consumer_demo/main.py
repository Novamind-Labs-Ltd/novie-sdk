"""Minimal A2A agent demonstrating consumer-side usage of novie-agent-sdk.

Run:
    pip install -e .
    novie-demo
    # …or…
    python -m novie_python_consumer_demo.main
"""

from __future__ import annotations

from pathlib import Path

from novie_agent_sdk import (
    Agent,
    TaskContext,
    extract_project_brief,
    render_brief_for_prompt,
)


MANIFEST_PATH = Path(__file__).resolve().parents[2] / ".well-known" / "agent.json"


def build_agent() -> Agent:
    agent = Agent.from_manifest(MANIFEST_PATH)

    @agent.task
    async def handle(ctx: TaskContext) -> dict:
        brief = extract_project_brief(ctx.input)
        prompt_extra = render_brief_for_prompt(brief) if brief is not None else ""

        await ctx.emit_message("demo: received task")
        if prompt_extra:
            await ctx.emit_message(f"demo: project brief attached ({len(prompt_extra)} chars)")

        return {
            "echo": ctx.input,
            "brief_chars": len(prompt_extra),
            "status": "ok",
        }

    return agent


def run() -> None:
    build_agent().serve(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
