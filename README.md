# torch-mps-workarounds

Explicit PyTorch adapters for single-layer projected LSTMs on Apple Metal.
Keep your learned parameters, checkpoint keys, and projected hidden-state
interface while choosing an implementation that can execute on MPS.

MPS itself works with supported PyTorch operations. This package addresses the
specific projected-LSTM gap: the native MPS implementation in
[PyTorch v2.15.0](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mps/operations/RnnOps.mm#L91-L95)
still rejects projections. The adapters compose existing PyTorch operations;
they do not replace Metal or claim a new recurrent-network algorithm.

The current package provides two opt-in implementations of the standard
[projected LSTM equations](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.LSTM.html):

| Implementation | Recurrence | Intended use |
| --- | --- | --- |
| `ProjectedLSTM` | Ordinary tensor operations and an explicit time-step loop | Default implementation and correctness reference |
| `FusedProjectedLSTM` | Exact first step followed by a fused ordinary LSTM | Optional implementation to benchmark for your workload |

The package never patches PyTorch globally, chooses a device, or silently
moves unsupported work to the CPU. Performance depends on shapes, hardware,
and PyTorch version; a workaround can be slower than native CPU execution.

## Quick start

Use Python 3.10+ and an existing PyTorch installation. The only runtime
dependency is `torch`. From this checkout, add `PYTHONPATH=src`, or install
into an environment that already has PyTorch and setuptools:

```sh
python -m pip install --no-deps --no-build-isolation -e .
```

Patch explicitly, keep the returned root, then choose the device and optimizer:

```python
import torch
from torch_mps_workarounds import replace_projected_lstm

native = torch.nn.LSTM(16, 32, proj_size=8, batch_first=True)
checkpoint = native.state_dict()

model = replace_projected_lstm(native)
model.load_state_dict(checkpoint, strict=True)
model = model.to("mps")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(2, 17, 16, device="mps")
output, (hidden, cell) = model(x)
output.square().mean().backward()
optimizer.step()
```

For MPS execution, set `PYTORCH_ENABLE_MPS_FALLBACK=0` **before Python starts**.
The package does not alter this environment variable.

The [synthetic example](examples/projected_lstm.py) runs on CPU by default and
compares outputs, states, gradients, chunking, and an optimizer update with
native CPU `nn.LSTM`:

```sh
OMP_NUM_THREADS=1 PYTHONPATH=src \
  python examples/projected_lstm.py --device cpu

# Run this only when you intend to use the Metal GPU.
OMP_NUM_THREADS=1 PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python examples/projected_lstm.py --device mps --implementation fused
```

Add `--show-native-error` to the MPS command to first attempt the native
projected layer and print its actual result or error on that installation.

## Supported scope

Both adapters support:

- Exactly one layer, one direction, `batch_first=True`, and
  `0 < proj_size < hidden_size`.
- Dense input `(batch, time, input_size)` with nonzero time length.
- Bias enabled or disabled.
- Hidden state `(1, batch, proj_size)` and cell state
  `(1, batch, hidden_size)`.
- Autograd, optimizer updates, and chunking by passing returned state to
  the next call. Detach state explicitly when using truncated backpropagation.

Omitting state starts from zero on every call; the module does not retain a
hidden state between calls. Single-layer dropout has no effect, matching
native `nn.LSTM`.

CPU float32/float64 and MPS float32 have parity coverage. Other dtypes,
`PackedSequence`, unbatched input, multiple layers, bidirectional execution,
and compilation/export modes are outside the validated scope. Unsupported
projected module variants raise before the patcher changes the model tree.

## Parameters and checkpoints

`replace_projected_lstm(model)` recursively replaces supported projected
`nn.LSTM` modules and returns the resulting root. `ProjectedLSTM(layer)` also
wraps one layer directly.

Replacement preserves parameter objects, values, names, ordering, checkpoint
keys, and shared child aliases. Nonprojected LSTMs and other modules/buffers
remain untouched. Repeating the patch is safe. Choose an implementation before
patching; patchers leave already-adapted modules as they are.

Patch before moving devices and constructing an optimizer. An existing
optimizer keeps its parameter references during replacement, but a later device
move may replace those objects. Module-level hooks on the original native LSTM
are not transferred; attach hooks to the adapted module.

## Optional fused path and version risk

```python
from torch_mps_workarounds import replace_projected_lstm_fused

model = replace_projected_lstm_fused(native_model)
```

`FusedProjectedLSTM(layer)` is the corresponding single-layer constructor.

The effective recurrent matrix is `W_hh @ W_hr`. The fused adapter computes
the first step directly from the supplied **projected** hidden state and full
cell state, then uses an ordinary fused LSTM for the remaining steps. This
preserves the boundary when a detached state was produced before an optimizer
update changed the projection. Effective weights are recomputed with autograd
on each call; they are never cached or detached.

The fused tail calls private `torch._VF.lstm`. Its signature, autograd behavior,
and backend support can change with PyTorch releases. Saved GPU validation uses
PyTorch **2.14.1**. Re-run parity after an upgrade, and select the manual adapter
explicitly if the private call becomes incompatible. Both adapters preserve
native recurrent-module metadata and should be checked on the version you use.

## Tests and recorded evidence

The latest [validation receipt](evidence/release_actual_mps_validation.json) and
[test log](evidence/release_actual_mps_unittest.txt) record **27 passing synthetic
tests with zero skips** on an Apple M4 Pro, macOS 27.0.1, and PyTorch 2.14.1,
with CPU fallback disabled. This run includes **18 CPU tests and 9 actual-MPS
tests**, covering both implementations. The earlier
[6 November receipt](evidence/actual_mps_validation.json) is also preserved.

Coverage includes outputs, final states, input/state/parameter gradients,
chunking, reset behavior, strict checkpoint loading, aliases, optimizer steps,
single-step fused execution, and detached state carried across an optimizer
update. Tolerances are explicit in the test suite; these cases establish a
bounded compatibility result.

Run the CPU suite with one thread:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  TORCH_MPS_WORKAROUNDS_TEST_MPS=0 \
  python -m unittest discover -s tests -v
```

MPS tests are skipped in this mode. The GitHub Actions workflow is configured
for this CPU mode on Ubuntu and macOS; a passing CPU job does not establish GPU
support.

When the GPU is available for an explicit check:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  PYTORCH_ENABLE_MPS_FALLBACK=0 TORCH_MPS_WORKAROUNDS_TEST_MPS=1 \
  python -m unittest discover -s tests -v
```

An explicit MPS request fails if MPS is unavailable or fallback is not disabled.
A skipped MPS test is not a successful GPU check.

## Benchmark your workload

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python -m torch_mps_workarounds.benchmark \
    --implementation fused --warmup 2 --iterations 5 \
    --output local_mps_benchmark.json
```

The benchmark checks forward/backward parity, warms both implementations,
synchronizes Metal around timed phases, and records hardware, PyTorch version,
shapes, sample counts, and median forward/backward times against one-thread
native CPU execution. It measures the recurrent operation and its backward
pass, excluding optimizer updates and data loading. Use `--implementation manual`
to measure the default path. This README makes no general speedup claim.

A saved [M4 Pro measurement](evidence/release_m4pro_fused_benchmark.json) uses
batch 32, 1,000 steps, input size 168, hidden size 320, and projection size 128.
Forward and gradient parity passed before timing. The medians from five timed
iterations, after two warmups, were:

| Operation | Native CPU, one thread | Fused MPS |
| --- | ---: | ---: |
| Forward | 123.90 ms | 20.11 ms |
| Backward | 172.28 ms | 57.27 ms |

Reproduce this particular synthetic workload by adding `--batch 32 --steps 1000`
to the fused benchmark command above. These measurements do not establish
end-to-end training speed or performance for other shapes.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for bounded CPU checks, scheduled MPS
validation, and the information needed for a reproducible report. Contributions
should preserve documented semantics and credit earlier work they build on.

Maintained by [Japjot Singh](https://github.com/japjotsingh-star).
Developed with AI coding assistance. Licensed under the [MIT license](LICENSE).
