"""Explicit PyTorch workarounds; importing does not patch PyTorch globally.

Names are imported lazily so that ``enable_mps_fallback`` can run before torch
is first imported, which PyTorch requires for the fallback flag to take effect.
"""

from importlib import import_module

from .env import enable_mps_fallback

_LAZY = {
    "ProjectedLSTM": "projected_lstm", "replace_projected_lstm": "projected_lstm",
    "FusedProjectedLSTM": "fused_projected_lstm", "replace_projected_lstm_fused": "fused_projected_lstm",
    "FallbackGRU": "gru", "replace_gru": "gru",
}
__all__ = [*_LAZY, "enable_mps_fallback"]
__version__ = "0.1.0"


def __getattr__(name):
    if name in _LAZY:
        value = getattr(import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
