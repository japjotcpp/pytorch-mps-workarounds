"""GRU adapter parity with native CPU nn.GRU; MPS execution is explicitly opt-in."""

import copy
import os
import unittest

import torch
from torch import nn

from torch_mps_workarounds import FallbackGRU, replace_gru

CONFIGS = (
    {},
    {"num_layers": 2},
    {"bidirectional": True},
    {"num_layers": 3, "bidirectional": True},
    {"bias": False, "num_layers": 2, "bidirectional": True},
    {"batch_first": False, "num_layers": 2, "bidirectional": True},
)


def setUpModule():
    torch.set_num_threads(1)


def run(module, x, h, probe_out, probe_h):
    x, h = x.clone().requires_grad_(), h.clone().requires_grad_()
    module.zero_grad()
    out, hn = module(x, h)
    ((out * probe_out).sum() + (hn * probe_h).sum()).backward()
    grads = [x.grad, h.grad] + [p.grad.clone() for p in module.parameters()]
    return [out.detach(), hn.detach()] + [g.detach() for g in grads]


class Checks:
    device = "cpu"
    dtype = torch.float64

    def tol(self):
        if self.device == "mps":
            return 1e-5, 2e-3
        return (1e-11, 1e-8) if self.dtype == torch.float64 else (3e-6, 3e-4)

    def check(self, **options):
        torch.manual_seed(7)
        options = {"batch_first": True, **options}
        native = nn.GRU(6, 5, **options).to(self.dtype)
        layout = (2, 9) if options["batch_first"] else (9, 2)
        directions = 2 if options.get("bidirectional") else 1
        x = torch.randn(*layout, 6, dtype=self.dtype) * 0.5
        h = torch.randn(options.get("num_layers", 1) * directions, 2, 5, dtype=self.dtype) * 0.3
        out_shape = (*layout, 5 * directions)
        probes = torch.randn(out_shape, dtype=self.dtype), torch.randn(h.shape, dtype=self.dtype)
        adapter = replace_gru(copy.deepcopy(native)).to(self.device)
        self.assertIsInstance(adapter, FallbackGRU)
        expected = run(native, x, h, *probes)
        actual = run(adapter, *(t.to(self.device) for t in (x, h, *probes)))
        atol, rtol = self.tol()
        self.assertEqual(len(expected), len(actual))
        for a, b in zip(expected, actual):
            torch.testing.assert_close(b.cpu(), a, atol=atol, rtol=rtol)
        # default zero state matches native
        with torch.no_grad():
            torch.testing.assert_close(
                adapter(x.to(self.device))[0].cpu(), native(x)[0], atol=atol, rtol=rtol)

    def test_parity(self):
        for options in CONFIGS:
            with self.subTest(options=options):
                self.check(**options)


class TestCPU(Checks, unittest.TestCase):
    def test_float32(self):
        self.dtype = torch.float32
        self.check(num_layers=2, bidirectional=True)

    def test_dropout_eval_and_train(self):
        native = nn.GRU(6, 5, num_layers=2, dropout=0.5, batch_first=True)
        x = torch.randn(2, 9, 6)
        adapter = FallbackGRU(copy.deepcopy(native))
        native.eval(), adapter.eval()
        torch.testing.assert_close(adapter(x)[0], native(x)[0], atol=3e-6, rtol=3e-4)
        adapter.train()
        a, b = adapter(x)[0], adapter(x)[0]
        self.assertFalse(torch.equal(a, b))  # inter-layer dropout is active

    def test_replacement_preserves_parameters_and_aliases(self):
        gru = nn.GRU(4, 3, 2, batch_first=True, bidirectional=True)
        root = nn.Module()
        root.a, root.b, root.lstm, root.norm = gru, gru, nn.LSTM(4, 3), nn.LayerNorm(4)
        before = dict(root.named_parameters())
        keys = list(root.state_dict())
        lstm, norm = root.lstm, root.norm
        self.assertIs(replace_gru(root), root)
        self.assertIs(root.a, root.b)
        self.assertIsInstance(root.a, FallbackGRU)
        self.assertIs(root.lstm, lstm)
        self.assertIs(root.norm, norm)
        self.assertEqual(keys, list(root.state_dict()))
        self.assertTrue(all(before[n] is p for n, p in root.named_parameters()))
        adapted = root.a
        replace_gru(root)
        self.assertIs(root.a, adapted)
        self.assertIsNone(adapted.flatten_parameters())
        self.assertTrue(all(a is b for a, b in zip(adapted._flat_weights, gru._flat_weights)))

    def test_root_replacement_and_errors(self):
        self.assertIsInstance(replace_gru(nn.GRU(4, 3)), FallbackGRU)
        with self.assertRaises(TypeError):
            FallbackGRU(nn.LSTM(4, 3))
        with self.assertRaises(TypeError):
            replace_gru(None)
        m = FallbackGRU(nn.GRU(4, 3, batch_first=True))
        for bad in (None, torch.zeros(2, 4), torch.zeros(2, 0, 4), torch.zeros(2, 5, 3)):
            with self.assertRaises(ValueError):
                m(bad)
        with self.assertRaises(ValueError):
            m(torch.zeros(2, 5, 4), torch.zeros(2, 2, 3))


@unittest.skipUnless(os.environ.get("TORCH_MPS_WORKAROUNDS_TEST_MPS") == "1", "actual MPS tests require explicit opt-in")
class TestMPS(Checks, unittest.TestCase):
    device = "mps"
    dtype = torch.float32

    @classmethod
    def setUpClass(cls):
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise RuntimeError("MPS tests require PYTORCH_ENABLE_MPS_FALLBACK=0 before Python starts")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable; GPU checks did not run")


if __name__ == "__main__":
    unittest.main()
