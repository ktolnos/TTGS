"""Bridge module exposing SAW agent config to ml_collections config_flags.

The original SAW implementation lives under impls/saw/, which is vendored as a
submodule. This shim loads its `get_config` lazily so we can reference it via
`--agent=agents/saw.py` like the rest of the agents in this repo.
"""

import importlib.util
from pathlib import Path
from typing import Any


def _load_vendor_module():
    repo_root = Path(__file__).resolve().parents[2]
    saw_agent_path = repo_root / "impls" / "saw" / "impls" / "agents" / "saw.py"
    if not saw_agent_path.exists():
        raise ImportError(f"SAW agent source not found at {saw_agent_path}")
    module_name = "ttgs_vendor_saw_config"
    spec = importlib.util.spec_from_file_location(module_name, saw_agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load SAW agent spec from {saw_agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SAW_MODULE = _load_vendor_module()


def get_config() -> Any:
    """Return the SAW agent config (ml_collections.ConfigDict)."""
    return _SAW_MODULE.get_config()


SAWAgent = _SAW_MODULE.SAWAgent
