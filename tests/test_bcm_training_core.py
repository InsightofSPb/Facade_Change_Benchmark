import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

from facade_change.methods.bcm_training_core import bcm_bits_map, bcm_nll, _logistic_mixture_bits


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
class BCMTrainingCoreTests(unittest.TestCase):
    def setUp(self):
        import torch
        from torch import nn

        self.torch = torch

        class Features(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.2))
                self.seen = []

            def forward(self, plane):
                self.seen.append(None if plane is None else plane.detach().clone())
                return None if plane is None else plane[:, :, ::2, ::2] * self.weight

        class Context(nn.Module):
            def forward(self, lossy_feats, forward_ref_feats=None, backward_ref_feats=None):
                if forward_ref_feats is None:
                    return lossy_feats
                return lossy_feats + forward_ref_feats + backward_ref_feats

        class AutoContext(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.1))
                self.seen = []

            def forward(self, values):
                self.seen.append(values.detach().clone())
                return values.sum(dim=1, keepdim=True) * self.weight

        class Parameters(nn.Module):
            def __init__(self, index):
                super().__init__()
                self.mean = nn.Parameter(torch.tensor(float(index)))
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.seen = []

            def forward(self, context):
                self.seen.append(context.detach().clone())
                mean = context.sum(dim=1, keepdim=True) + self.mean
                return torch.cat([torch.zeros_like(mean), mean,
                                  torch.zeros_like(mean) + self.scale], dim=1)

        class Entropy(nn.Module):
            def __init__(self):
                super().__init__()
                self.auto_ctx_extraction = nn.ModuleList([AutoContext() for _ in range(3)])
                self.params_estimator = nn.ModuleList([Parameters(index) for index in range(4)])
                self.discrete_logistic_mixture_model = type("Mixtures", (), {"K": 1})()

            @staticmethod
            def spatial_split(plane):
                return (plane[:, :, ::2, ::2], plane[:, :, 1::2, 1::2],
                        plane[:, :, ::2, 1::2], plane[:, :, 1::2, ::2])

            @staticmethod
            def spatial_merge(phases):
                n, c, h, w = phases[0].shape
                result = phases[0].new_zeros(n, c, 2 * h, 2 * w)
                for phase, (row, col) in zip(phases, [(0, 0), (1, 1), (0, 1), (1, 0)]):
                    result[:, :, row::2, col::2] = phase
                return result

        class Network(nn.Module):
            bit_depth = 8

            def __init__(self):
                super().__init__()
                self.feats_extract_lossy_rec = Features()
                self.feats_extract_forward_ref = Features()
                self.feats_extract_backward_ref = Features()
                self.i2cg = Context()
                self.entropy_models = Entropy()

            def compress(self, *args, **kwargs):
                raise AssertionError("Training must not call inference/entropy coding")

        self.network = Network()
        self.residues = torch.tensor([[[[-255., 1., -2., 3.], [4., 255., -6., 7.],
                                       [-8., 9., -10., 11.], [12., -13., 14., -15.]]]])
        self.base = torch.full_like(self.residues, 128.)
        self.base[0, 0, 0, 0] = 255.
        self.base[0, 0, 1, 1] = 0.
        self.reference = torch.full_like(self.base, 255.)

    def test_gradient_and_bits_map_reduction(self):
        torch = self.torch
        bits = bcm_bits_map(self.network, self.residues, self.base, self.reference)
        self.assertEqual(bits.shape, self.residues.shape)
        self.assertTrue(torch.isfinite(bits).all())
        torch.testing.assert_close(bcm_nll(self.network, self.residues, self.base,
                                         self.reference), bits.sum(dim=(1, 2, 3)))
        bits.sum().backward()
        for name, parameter in self.network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)

    def test_normalization_and_no_reference_branch(self):
        torch = self.torch
        bcm_nll(self.network, self.residues, self.base, self.reference)
        torch.testing.assert_close(self.network.feats_extract_lossy_rec.seen[-1], self.base / 255.)
        for name in ("feats_extract_forward_ref", "feats_extract_backward_ref"):
            torch.testing.assert_close(getattr(self.network, name).seen[-1], self.reference / 255.)
        bcm_nll(self.network, self.residues, self.base)
        self.assertIsNone(self.network.feats_extract_forward_ref.seen[-1])
        self.assertIsNone(self.network.feats_extract_backward_ref.seen[-1])

    def test_only_preceding_phases_enter_context(self):
        torch = self.torch
        entropy = self.network.entropy_models
        bcm_nll(self.network, self.residues, self.base, self.reference)
        phases = entropy.spatial_split(self.residues)
        contexts = [module.seen[-1].clone() for module in entropy.params_estimator]
        for index, module in enumerate(entropy.auto_ctx_extraction, start=1):
            torch.testing.assert_close(module.seen[-1], torch.cat(phases[:index], dim=1))
        # Changing the current/final phase cannot influence any estimated
        # distribution. Its target likelihood changes, while contexts do not.
        changed = self.residues.clone()
        changed[:, :, 1::2, ::2] += 1.
        bcm_nll(self.network, changed, self.base, self.reference)
        for before, module in zip(contexts, entropy.params_estimator):
            torch.testing.assert_close(before, module.seen[-1])
        # Changing phase 2 (BR) affects later phases, not phases 1 or 2.
        changed = self.residues.clone()
        changed[:, :, 1::2, 1::2] -= 1.
        bcm_nll(self.network, changed, self.base, self.reference)
        for index, (before, module) in enumerate(zip(contexts, entropy.params_estimator)):
            if index < 2:
                torch.testing.assert_close(before, module.seen[-1])
            else:
                self.assertFalse(torch.equal(before, module.seen[-1]))

    def test_logistic_mass_matches_analytic_and_extreme_values(self):
        torch = self.torch
        target = torch.tensor([[[[0.]]]], dtype=torch.float64)
        params = torch.zeros((1, 3, 1, 1), dtype=torch.float64, requires_grad=True)
        expected = -math.log2(1 / (1 + math.exp(-0.5)) - 1 / (1 + math.exp(0.5)))
        self.assertAlmostEqual(float(_logistic_mixture_bits(params, target, 1)), expected, places=12)
        params = torch.tensor([math.log(0.2), math.log(0.8), -1., 2.,
                               math.log(0.5), math.log(3.)], dtype=torch.float64).reshape(1, 6, 1, 1)
        params.requires_grad_()
        target = torch.ones((1, 1, 1, 1), dtype=torch.float64)
        probability = sum(weight * (1 / (1 + math.exp(-(1.5 - mean) / scale))
                                    - 1 / (1 + math.exp(-(0.5 - mean) / scale)))
                          for weight, mean, scale in [(0.2, -1., 0.5), (0.8, 2., 3.)])
        bits = _logistic_mixture_bits(params, target, 2)
        self.assertAlmostEqual(float(bits), -math.log2(probability), places=12)
        bits.sum().backward()
        self.assertTrue((params.grad.abs() > 0).all())
        # CDF subtraction would round to zero here. Stable log likelihood and
        # its gradients remain finite, including a scale that exp underflows.
        params = torch.tensor([[[[1000., -1000., 0.]], [[1000., -1000., 0.]],
                                [[-100., 50., 1000.]]]], requires_grad=True)
        target = torch.tensor([[[[-255., 255., 0.]]]])
        bits = _logistic_mixture_bits(params, target, 1)
        self.assertTrue(torch.isfinite(bits).all())
        bits.sum().backward()
        self.assertTrue(torch.isfinite(params.grad).all())
        self.assertGreater(float(params.grad[:, 1, :, 0].abs().sum()), 0.)
        self.assertGreater(float(params.grad[:, 2, :, 2].abs().sum()), 0.)

    def test_rejects_lossy_or_wrapped_inputs(self):
        torch = self.torch
        invalid = [self.residues + 0.25, self.residues * float("nan"),
                   self.residues.clone(), self.residues[:, :, :3],
                   self.residues.repeat(1, 3, 1, 1)]
        invalid[2][0, 0, 0, 0] = -256.
        for value in invalid:
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                bcm_nll(self.network, value, self.base, self.reference)
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            bcm_nll(self.network, torch.full_like(self.residues, 255.), self.base)
        with self.assertRaisesRegex(ValueError, "reference"):
            bcm_nll(self.network, self.residues, self.base,
                    torch.full_like(self.reference, 128.) / 255.)
        self.network.bit_depth = 16
        with self.assertRaisesRegex(ValueError, "bit_depth"):
            bcm_nll(self.network, self.residues, self.base)


@unittest.skipUnless(os.environ.get("FACADE_BCM_SOURCE") and importlib.util.find_spec("torch")
                     and importlib.util.find_spec("einops"),
                     "Set FACADE_BCM_SOURCE to the original author checkout")
class OriginalBCMTrainingCoreTests(unittest.TestCase):
    def test_original_network_backward_without_entropy_coder(self):
        # Isolate the author's generic Modules namespace. The coder stub raises
        # if used; training needs its nn modules, not the torchac C++ extension.
        script = """
import importlib.util
import os
from pathlib import Path
import sys
import types
import torch
from facade_change.methods.bcm_training_core import bcm_nll
torch.set_num_threads(1)
source = Path(os.environ['FACADE_BCM_SOURCE']).resolve()
sys.path.insert(0, str(source))
def forbidden(*args, **kwargs):
    raise AssertionError('Training invoked the arithmetic coder')
stub = types.ModuleType('torchac')
stub.torchac = types.SimpleNamespace(encode_float_cdf=forbidden, decode_float_cdf=forbidden)
sys.modules['torchac'] = stub
spec = importlib.util.spec_from_file_location('original_bcm_training_test', source / 'Network.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
network = module.Network(bit_depth=8).train()
base = torch.full((1, 1, 16, 16), 128.)
residues = (torch.arange(256).reshape(1, 1, 16, 16) % 7 - 3).float()
reference = torch.full_like(base, 127.)
loss = bcm_nll(network, residues, base, reference).mean() / 256
assert torch.isfinite(loss)
loss.backward()
groups = ['feats_extract_lossy_rec', 'feats_extract_forward_ref',
          'feats_extract_backward_ref', 'i2cg', 'entropy_models']
for group in groups:
    grads = [param.grad for param in getattr(network, group).parameters() if param.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads), group
    assert sum(float(grad.abs().sum()) for grad in grads) > 0, group
print('Original BCM gradient smoke passed')
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
