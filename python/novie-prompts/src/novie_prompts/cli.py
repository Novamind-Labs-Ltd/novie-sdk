"""`novie-prompts sync push` — seed each prompt at `production` from the repo
constant, CREATE-IF-ABSENT only. NEVER overwrites an existing production version
(D2: don't clobber the live source of truth with the stale replica)."""
from __future__ import annotations
import sys
from langfuse.api import NotFoundError


def sync_push(prompts: dict[str, str], *, client) -> list[str]:
    created: list[str] = []
    for name, text in prompts.items():
        try:
            client.get_prompt(name, label="production")
            continue                       # exists → never overwrite
        except NotFoundError:
            pass
        client.create_prompt(name=name, prompt=text, labels=["production"])
        created.append(name)
    return created


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv[:2] != ["sync", "push"]:
        print("usage: novie-prompts sync push", file=sys.stderr)
        return 2
    # Real wiring: the consumer repo passes its prompt dict + a configured client.
    # (Consumer-specific; not exercised here — sync_push carries the logic + tests.)
    print("novie-prompts sync push: invoke sync_push(prompts, client=...) from the consumer")
    return 0
