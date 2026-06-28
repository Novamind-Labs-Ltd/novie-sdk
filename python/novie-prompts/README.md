# novie-prompts

Fail-soft Langfuse-managed prompt fetch with in-repo fallback (ADR-075). **Content prompts only** — control-plane prompts stay in-code constants and are never fetched (ADR-075 D4).

## Install

```bash
pip install "git+https://github.com/Novamind-Labs-Ltd/novie-sdk@<tag>#subdirectory=python/novie-prompts"
```

## Use

At boot, after resolving your Langfuse secrets, configure the connection and wire a metrics recorder:

```python
import novie_prompts

novie_prompts.configure(host=HOST, public_key=PK, secret_key=SK)
novie_prompts.set_recorder(my_rcp_metrics_recorder)  # has record_fallback(name, reason) + record_live(name)
assert novie_prompts.has_recorder()  # fail loud at boot if telemetry is unwired
```

Then resolve a **content** prompt, always passing the in-repo constant as the fallback:

```python
from .prompts import _ANALYST_SYSTEM_PROMPT

system_prompt = novie_prompts.get_managed_prompt("analyst/system", fallback=_ANALYST_SYSTEM_PROMPT)
```

`get_managed_prompt` NEVER raises and is latency-bounded. It returns the constant when Langfuse is disabled (`NOVIE_OBSERVABILITY_LANGFUSE_ENABLED=false`), unreachable, slow, missing, or chat-type, recording a per-reason counter (`disabled | timeout | missing | chat_type | exception`) on every exit.

## Testing (no Langfuse in CI)

```python
from novie_prompts import testing

def test_my_consumer():
    fake, rec = testing.install_fake(text="LIVE BODY")
    # ... call code that calls get_managed_prompt ...
    testing.reset()
```
