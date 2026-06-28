import importlib


def test_package_imports():
    mod = importlib.import_module("novie_prompts")
    assert mod is not None
