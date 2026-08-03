import os

import yaml

CONFIG_PATH = "config/darshan_config.yaml"


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to load config: {e}")
            return {}


def get_config_val(config, path, default=None):
    keys = path.split(".")
    val = config
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val


darshan_config = load_config()
import platform

import pennylane as qml

_cached_backend = None


def get_optimal_backend():
    global _cached_backend
    if _cached_backend is not None:
        return _cached_backend
    os_name = platform.system().lower()
    if os_name == "linux":
        try:
            qml.device("lightning.gpu", wires=1)
            _cached_backend = "lightning.gpu"
            return _cached_backend
        except Exception:
            pass
    try:
        qml.device("lightning.qubit", wires=1)
        _cached_backend = "lightning.qubit"
        return _cached_backend
    except Exception:
        pass
    _cached_backend = "default.qubit"
    return _cached_backend
