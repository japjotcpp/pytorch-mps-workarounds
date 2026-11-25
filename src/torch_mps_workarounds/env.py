"""Process-level MPS fallback switch (opt-in, never called on import)."""

import os
import sys


def enable_mps_fallback() -> None:
    """Set PYTORCH_ENABLE_MPS_FALLBACK=1 so unsupported MPS ops run on the CPU.

    Call this BEFORE ``import torch`` (probed on PyTorch 2.15.0: setting it
    after torch is imported has no effect, even before any MPS work). This
    package's ``__init__`` does not import torch, so
    ``from torch_mps_workarounds import enable_mps_fallback`` is safe first. Fallback ops are slow (a
    device-to-host copy each) and warn once; this conflicts with the explicit
    PYTORCH_ENABLE_MPS_FALLBACK=0 the rest of this package's MPS tests require.
    """
    if "torch" in sys.modules:
        import warnings
        warnings.warn("torch is already imported; the fallback flag will not take effect")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
