"""Bounded synthetic actual-MPS benchmark with a native CPU reference."""

import argparse
import copy
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
from torch import nn

from .projected_lstm import ProjectedLSTM
from .fused_projected_lstm import FusedProjectedLSTM


def _synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def _run(module, tensors):
    module.zero_grad(set_to_none=True)
    x, h, c = [value.detach().clone().requires_grad_(True) for value in tensors]
    _synchronize(x.device)
    started = time.perf_counter()
    output, state = module(x, (h, c))
    loss = sum(value.square().mean() for value in (output, *state))
    _synchronize(x.device)
    forward = time.perf_counter() - started
    started = time.perf_counter()
    loss.backward()
    _synchronize(x.device)
    backward = time.perf_counter() - started
    gradients = (x.grad, h.grad, c.grad, *(parameter.grad for parameter in module.parameters()))
    if not all(value is not None and torch.isfinite(value).all() for value in gradients):
        raise RuntimeError("benchmark produced missing or nonfinite gradients")
    return {
        "forward_seconds": forward,
        "backward_seconds": backward,
        "values": tuple(value.detach().cpu() for value in (output, *state, loss)),
        "gradients": tuple(value.detach().cpu().clone() for value in gradients),
    }


def _measure(module, tensors, warmup, iterations):
    for _ in range(warmup):
        _run(module, tensors)
    records = [_run(module, tensors) for _ in range(iterations)]
    return {
        key: {
            "median": statistics.median(record[key] for record in records),
            "min": min(record[key] for record in records),
        }
        for key in ("forward_seconds", "backward_seconds")
    }


def _hardware():
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], check=True,
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--input-size", type=int, default=168)
    parser.add_argument("--hidden-size", type=int, default=320)
    parser.add_argument("--proj-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--implementation", choices=("manual", "fused"), default="manual")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(args.batch, args.steps, args.input_size, args.hidden_size, args.proj_size, args.iterations) < 1:
        parser.error("dimensions and iterations must be positive")
    if args.warmup < 1 or args.proj_size >= args.hidden_size:
        parser.error("warmup must be positive and proj-size must be less than hidden-size")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("set PYTORCH_ENABLE_MPS_FALLBACK=0 before starting Python")
    if not torch.backends.mps.is_available():
        parser.error("actual MPS is unavailable; no GPU benchmark was run")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261106)
    native = nn.LSTM(args.input_size, args.hidden_size, proj_size=args.proj_size, batch_first=True)
    implementation = ProjectedLSTM if args.implementation == "manual" else FusedProjectedLSTM
    manual = implementation(copy.deepcopy(native)).to("mps")
    tensors = (
        torch.randn(args.batch, args.steps, args.input_size) * 0.2,
        torch.randn(1, args.batch, args.proj_size) * 0.1,
        torch.randn(1, args.batch, args.hidden_size) * 0.1,
    )
    moved = tuple(value.to("mps") for value in tensors)
    reference, actual = _run(native, tensors), _run(manual, moved)
    max_abs = {}
    for kind in ("values", "gradients"):
        errors = []
        for expected, value in zip(reference[kind], actual[kind]):
            torch.testing.assert_close(value, expected, atol=1e-5, rtol=2e-3)
            errors.append((value.double() - expected.double()).abs().max().item())
        max_abs[kind] = max(errors)
    report = {
        "torch": torch.__version__, "platform": platform.platform(), "hardware": _hardware(),
        "device": "mps", "implementation": args.implementation,
        "dtype": "torch.float32", "threads": torch.get_num_threads(),
        "fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "shape": [args.batch, args.steps, args.input_size, args.hidden_size, args.proj_size],
        "warmup": args.warmup, "iterations": args.iterations, "synthetic_only": True,
        "parity": {"status": "PASS", "atol": 1e-5, "rtol": 2e-3, "max_abs": max_abs},
        "timing_seconds": {
            "cpu_native": _measure(native, tensors, args.warmup, args.iterations),
            "mps_" + args.implementation: _measure(manual, moved, args.warmup, args.iterations),
        },
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
