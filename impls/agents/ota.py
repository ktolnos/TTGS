"""Bridge module exposing OTA agent config to ml_collections config_flags.

This shim loads the vendored OTA implementation under impls/ota-v so it can be
referenced via `--agent=agents/ota.py` like other agents in this repo.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_vendor_module():
    repo_root = Path(__file__).resolve().parents[2]
    ota_root = repo_root / "impls" / "ota-v"
    ota_agent_path = ota_root / "agents" / "ota.py"
    if not ota_agent_path.exists():
        raise ImportError(f"OTA agent source not found at {ota_agent_path}")
    # Ensure ota-v utilities take precedence over repo-level utils.
    ota_root_str = str(ota_root)
    if ota_root_str not in sys.path:
        sys.path.insert(0, ota_root_str)
    module_name = "ttgs_vendor_ota_config"
    spec = importlib.util.spec_from_file_location(module_name, ota_agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load OTA agent spec from {ota_agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OTA_MODULE = _load_vendor_module()


def get_config() -> Any:
    """Return the OTA agent config (ml_collections.ConfigDict)."""
    return _OTA_MODULE.get_config()


OTAAgent = _OTA_MODULE.OTAAgent
