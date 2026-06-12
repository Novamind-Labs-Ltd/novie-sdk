from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_base_package_import_does_not_require_langchain_core() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class BlockLangChain(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "langchain_core" or fullname.startswith("langchain_core."):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockLangChain())
import novie_agent_sdk
print(novie_agent_sdk.__name__)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "novie_agent_sdk"

