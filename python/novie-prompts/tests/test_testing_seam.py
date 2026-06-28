from novie_prompts import registry
from novie_prompts.testing import fake_registry


def test_force_timeout_falls_back():
    with fake_registry(mode="timeout"):
        assert registry.get_managed_prompt("p", fallback="FB") == "FB"
