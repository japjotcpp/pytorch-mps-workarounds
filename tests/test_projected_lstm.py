"""Synthetic parity checks; MPS execution is explicitly opt-in."""

import copy
import os
import unittest

import torch
from torch import nn

from torch_mps_workarounds import (
    FusedProjectedLSTM, ProjectedLSTM,
    replace_projected_lstm, replace_projected_lstm_fused,
)


def setUpModule():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def sample(batch=2, steps=17, inputs=16, hidden=32, projected=8, dtype=torch.float64):
    torch.manual_seed(20261106)
    native = nn.LSTM(inputs, hidden, proj_size=projected, batch_first=True).to(dtype)
    x = torch.randn(batch, steps, inputs, dtype=dtype) * 0.2
    h = torch.randn(1, batch, projected, dtype=dtype) * 0.1
    c = torch.randn(1, batch, hidden, dtype=dtype) * 0.1
    probes = tuple(torch.randn(*shape, dtype=dtype) for shape in ((batch, steps, projected), h.shape, c.shape))
    return native, (x, h, c), probes


def run(module, tensors, probes, chunks=None):
    module.zero_grad(set_to_none=True)
    x, h, c = [value.detach().clone().requires_grad_(True) for value in tensors]
    if chunks is None:
        output, state = module(x, (h, c))
    else:
        pieces, state, offset = [], (h, c), 0
        for length in chunks:
            piece, state = module(x[:, offset:offset + length], state)
            pieces.append(piece)
            offset += length
        if offset != x.shape[1]:
            raise AssertionError("chunks must cover the entire sequence")
        output = torch.cat(pieces, dim=1)
    values = (output, *state)
    loss = sum((value * probe).sum() / value.numel() for value, probe in zip(values, probes))
    loss.backward()
    gradients = {"input": x.grad, "initial_h": h.grad, "initial_c": c.grad}
    gradients.update({name: parameter.grad for name, parameter in module.named_parameters()})
    if not all(value is not None and torch.isfinite(value).all() for value in gradients.values()):
        raise AssertionError("all inputs, states, and parameters need finite gradients")
    return tuple(value.detach().clone() for value in (*values, loss)), {
        name: value.detach().clone() for name, value in gradients.items()
    }


class NumericalChecks:
    device = "cpu"
    implementation = ProjectedLSTM
    patch = staticmethod(replace_projected_lstm)

    def assert_values(self, reference, actual, dtype=torch.float64):
        atol, rtol = (1e-11, 1e-8) if dtype == torch.float64 else (3e-6, 3e-4)
        if self.device == "mps":
            atol, rtol = 1e-5, 2e-3
        torch.testing.assert_close(actual.detach().cpu(), reference.detach().cpu(), atol=atol, rtol=rtol)

    def assert_runs(self, reference, actual, dtype):
        for expected, result in zip(reference[0], actual[0]):
            self.assert_values(expected, result, dtype)
        self.assertEqual(reference[1].keys(), actual[1].keys())
        for name in reference[1]:
            self.assert_values(reference[1][name], actual[1][name], dtype)

    def device_sample(self, native, tensors, probes):
        return (
            self.implementation(copy.deepcopy(native)).to(self.device),
            tuple(value.to(self.device) for value in tensors),
            tuple(value.to(self.device) for value in probes),
        )

    def check_case(self, dtype, shape, bias=True):
        native, tensors, probes = sample(*shape, dtype=dtype)
        if not bias:
            native = nn.LSTM(shape[2], shape[3], proj_size=shape[4], batch_first=True, bias=False).to(dtype)
        manual, moved, moved_probes = self.device_sample(native, tensors, probes)
        reference = run(native, tensors, probes)
        actual = run(manual, moved, moved_probes)
        self.assertEqual(next(manual.parameters()).device.type, self.device)
        self.assert_runs(reference, actual, dtype)
        chunks = (1, 4, shape[1] - 5)
        chunked = run(manual, moved, moved_probes, chunks)
        self.assert_runs(reference, chunked, dtype)
        self.assert_runs(actual, chunked, dtype)

    def check_reset(self, dtype):
        native, tensors, probes = sample(dtype=dtype)
        manual, (x, h, c), _ = self.device_sample(native, tensors, probes)
        with torch.no_grad():
            expected, expected_state = native(tensors[0])
            first, first_state = manual(x)
            carried, _ = manual(x, (h, c))
            reset, reset_state = manual(x)
            zero, zero_state = manual(x, (torch.zeros_like(h), torch.zeros_like(c)))
        self.assertFalse(torch.equal(first, carried))
        for result in (first, reset, zero):
            self.assert_values(expected, result, dtype)
        for state in (first_state, reset_state, zero_state):
            for expected_value, value in zip(expected_state, state):
                self.assert_values(expected_value, value, dtype)

    def check_optimizer(self, dtype):
        native, tensors, probes = sample(dtype=dtype)
        manual, moved, moved_probes = self.device_sample(native, tensors, probes)
        reference_optimizer = torch.optim.SGD(native.parameters(), lr=0.1, momentum=0.9)
        actual_optimizer = torch.optim.SGD(manual.parameters(), lr=0.1, momentum=0.9)
        for _ in range(3):
            self.assert_runs(run(native, tensors, probes), run(manual, moved, moved_probes), dtype)
            reference_optimizer.step()
            actual_optimizer.step()
            for expected, result in zip(native.parameters(), manual.parameters()):
                self.assert_values(expected, result, dtype)
                self.assert_values(
                    reference_optimizer.state[expected]["momentum_buffer"],
                    actual_optimizer.state[result]["momentum_buffer"], dtype,
                )

    def check_carry_after_optimizer_update(self, dtype):
        native, tensors, probes = sample(dtype=dtype)
        manual, moved, moved_probes = self.device_sample(native, tensors, probes)
        expected, actual = run(native, tensors, probes), run(manual, moved, moved_probes)
        self.assert_runs(expected, actual, dtype)
        old_projection = native.weight_hr_l0.detach().clone()
        for module in (native, manual):
            optimizer = torch.optim.SGD(module.parameters(), lr=1.0)
            optimizer.step()
        self.assertFalse(torch.equal(old_projection, native.weight_hr_l0))
        # Retain detached OLD projected state after the optimizer changes W_hr.
        reference_tensors = (tensors[0], *expected[0][1:3])
        actual_tensors = (moved[0], *actual[0][1:3])
        self.assert_runs(
            run(native, reference_tensors, probes),
            run(manual, actual_tensors, moved_probes), dtype,
        )


class TestCPU(NumericalChecks, unittest.TestCase):
    def test_forward_backward_and_chunks(self):
        for dtype in (torch.float64, torch.float32):
            for bias in (True, False):
                with self.subTest(dtype=dtype, bias=bias):
                    self.check_case(dtype, (2, 17, 16, 32, 8), bias)

    def test_research_stream_shapes(self):
        for shape in ((1, 50, 168, 320, 128), (1, 50, 136, 128, 64)):
            with self.subTest(shape=shape):
                self.check_case(torch.float32, shape)

    def test_state_reset(self):
        self.check_reset(torch.float64)

    def test_optimizer_step(self):
        self.check_optimizer(torch.float64)

    def test_replacement_preserves_parameters_checkpoint_and_aliases(self):
        native, tensors, _ = sample()
        root = nn.Module()
        root.direct = native
        root.left = nn.Module()
        root.right = nn.Module()
        root.left.stream = native
        root.right.stream = native
        root.top = nn.LSTM(8, 16, batch_first=True)
        root.normalizer = nn.LayerNorm(8)
        root.register_buffer("mean", torch.randn(8))
        before = dict(root.named_parameters())
        checkpoint = copy.deepcopy(root.state_dict())
        original_top, original_normalizer, original_mean = root.top, root.normalizer, root.mean
        optimizer = torch.optim.Adam(root.parameters(), lr=0.001)
        native.eval()
        self.assertIs(self.patch(root), root)
        self.assertIs(root.direct, root.left.stream)
        self.assertIs(root.direct, root.right.stream)
        self.assertFalse(root.direct.training)
        self.assertIs(root.top, original_top)
        self.assertIs(root.normalizer, original_normalizer)
        self.assertIs(root.mean, original_mean)
        self.assertEqual(list(before), list(dict(root.named_parameters())))
        self.assertTrue(all(before[name] is parameter for name, parameter in root.named_parameters()))
        self.assertEqual(list(checkpoint), list(root.state_dict()))
        for name, value in checkpoint.items():
            self.assertTrue(torch.equal(value, root.state_dict()[name]))
        root.load_state_dict(checkpoint, strict=True)
        patched = root.direct
        self.assertIs(self.patch(root), root)
        self.assertIs(root.direct, patched)
        self.assertEqual({id(p) for p in before.values()}, {id(p) for group in optimizer.param_groups for p in group["params"]})
        optimizer.zero_grad(set_to_none=True)
        root.direct(tensors[0])[0].square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(root.direct.weight_ih_l0, checkpoint["direct.weight_ih_l0"]))
        fresh_native, _, _ = sample()
        fresh_native.load_state_dict(root.direct.state_dict(), strict=True)
        self.assert_values(fresh_native(tensors[0])[0], root.direct(tensors[0])[0])

    def test_root_replacement_and_flat_weights(self):
        native, _, _ = sample()
        manual = self.patch(native)
        self.assertIsInstance(manual, self.implementation)
        self.assertIs(manual.weight_ih_l0, native.weight_ih_l0)
        self.assertEqual(manual._flat_weights_names, native._flat_weights_names)
        self.assertTrue(all(a is b for a, b in zip(manual._flat_weights, native._flat_weights)))
        self.assertTrue(all(a is b for a, b in zip(manual.all_weights[0], native.all_weights[0])))
        self.assertIsNone(manual.flatten_parameters())
        self.assertIs(self.patch(manual), manual)

    def test_unsupported_variants_leave_tree_unchanged(self):
        variants = ({"num_layers": 2}, {"batch_first": False}, {"bidirectional": True})
        for options in variants:
            with self.subTest(options=options):
                root = nn.Module()
                root.good = nn.LSTM(4, 8, proj_size=2, batch_first=True)
                root.bad = nn.LSTM(4, 8, proj_size=2, **({"batch_first": True} | options))
                old_good, old_bad = root.good, root.bad
                with self.assertRaises(ValueError):
                    self.patch(root)
                self.assertIs(root.good, old_good)
                self.assertIs(root.bad, old_bad)
        native = nn.LSTM(4, 8, batch_first=True)
        self.assertIs(self.patch(native), native)
        with self.assertRaises(ValueError):
            self.implementation(native)
        with self.assertRaises(TypeError):
            self.implementation(nn.Linear(4, 8))
        with self.assertRaises(TypeError):
            self.patch(None)

    def test_invalid_inputs_and_states(self):
        native, (x, h, c), _ = sample()
        manual = self.implementation(native)
        for bad in (None, torch.zeros(17, 16), torch.zeros(2, 0, 16), torch.zeros(2, 17, 15)):
            with self.subTest(input=type(bad).__name__):
                with self.assertRaises(ValueError):
                    manual(bad)
        with self.assertRaises(ValueError):
            manual(x, (h.squeeze(0), c))
        with self.assertRaises(ValueError):
            manual(x, (h, c[..., :-1]))


@unittest.skipUnless(os.environ.get("TORCH_MPS_WORKAROUNDS_TEST_MPS") == "1", "actual MPS tests require explicit opt-in")
class TestMPS(NumericalChecks, unittest.TestCase):
    device = "mps"

    @classmethod
    def setUpClass(cls):
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise RuntimeError("MPS tests require PYTORCH_ENABLE_MPS_FALLBACK=0 before Python starts")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable; GPU checks did not run")

    def test_forward_backward_and_chunks(self):
        for bias in (True, False):
            with self.subTest(bias=bias):
                self.check_case(torch.float32, (2, 17, 16, 32, 8), bias)

    def test_research_stream_shapes(self):
        for shape in ((1, 50, 168, 320, 128), (1, 50, 136, 128, 64)):
            with self.subTest(shape=shape):
                self.check_case(torch.float32, shape)

    def test_state_reset(self):
        self.check_reset(torch.float32)

    def test_optimizer_step(self):
        self.check_optimizer(torch.float32)


class TestFusedCPU(TestCPU):
    implementation = FusedProjectedLSTM
    patch = staticmethod(replace_projected_lstm_fused)

    def test_single_step(self):
        native, tensors, probes = sample(steps=1)
        fused, moved, moved_probes = self.device_sample(native, tensors, probes)
        self.assert_runs(run(native, tensors, probes), run(fused, moved, moved_probes), torch.float64)

    def test_detached_state_after_optimizer_update(self):
        self.check_carry_after_optimizer_update(torch.float64)


class TestFusedMPS(TestMPS):
    implementation = FusedProjectedLSTM
    patch = staticmethod(replace_projected_lstm_fused)

    def test_detached_state_after_optimizer_update(self):
        self.check_carry_after_optimizer_update(torch.float32)


if __name__ == "__main__":
    unittest.main()
