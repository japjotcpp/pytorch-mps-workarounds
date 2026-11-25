"""enable_mps_fallback: env var is set; the fallback really works on MPS (opt-in)."""

import os
import subprocess
import sys
import unittest
from unittest import mock

from torch_mps_workarounds import enable_mps_fallback

CHILD = """
from torch_mps_workarounds import enable_mps_fallback
enable_mps_fallback()
import torch
x = torch.randint(0, 4, (6, 7)).float()
a, b = x.mode(1), x.to('mps').mode(1)  # aten::mode has no MPS kernel
assert torch.equal(a.values, b.values.cpu()), (a, b)
x = torch.randn(4, 4)
assert torch.allclose(torch.linalg.eigvals(x), torch.linalg.eigvals(x.to('mps')).cpu(), atol=1e-4)
print('ok')
"""


class TestEnv(unittest.TestCase):
    def test_sets_variable(self):
        with mock.patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}):
            with self.assertWarns(UserWarning):  # torch is already imported by the suite
                enable_mps_fallback()
            self.assertEqual(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"], "1")

    @unittest.skipUnless(os.environ.get("TORCH_MPS_WORKAROUNDS_TEST_MPS") == "1", "actual MPS tests require explicit opt-in")
    def test_fallback_matches_cpu_on_mps(self):
        env = {k: v for k, v in os.environ.items() if k != "PYTORCH_ENABLE_MPS_FALLBACK"}
        out = subprocess.run([sys.executable, "-W", "ignore", "-c", CHILD], env=env,
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("ok", out.stdout)


if __name__ == "__main__":
    unittest.main()
