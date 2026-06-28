from langfuse.api import NotFoundError
from novie_prompts.cli import sync_push


class FakeClient:
    def __init__(self, existing):
        self.existing = set(existing)
        self.created = []

    def get_prompt(self, name, **kw):
        if name not in self.existing:
            raise NotFoundError("nope")
        return object()

    def create_prompt(self, *, name, prompt, labels):
        assert labels == ["production"]
        self.created.append((name, prompt))
        self.existing.add(name)


def test_creates_absent_only():
    c = FakeClient(existing={"a"})
    created = sync_push({"a": "A-new", "b": "B"}, client=c)
    assert created == ["b"]                 # 'a' exists → skipped, NOT overwritten
    assert c.created == [("b", "B")]


def test_idempotent_second_run_is_noop():
    c = FakeClient(existing=set())
    sync_push({"a": "A"}, client=c)
    c.created.clear()
    created = sync_push({"a": "A"}, client=c)  # second run
    assert created == []
    assert c.created == []


def test_never_overwrites_when_constant_differs():
    c = FakeClient(existing={"a"})          # 'a' already in Langfuse
    created = sync_push({"a": "A-DIFFERENT"}, client=c)
    assert created == []                    # differing local constant does NOT push
    assert c.created == []
