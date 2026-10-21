"""CPU checks of public usage and masked input invariance, not paper accuracy."""
import unittest

import numpy as np
import torch

from cospec import build_model


class PublicModelTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.wave = np.linspace(400, 1000, 32, dtype=np.float32)
        self.means = np.zeros((3, 32), dtype=np.float32)
        self.variances = np.ones_like(self.means)

    def test_missing_values_do_not_affect_predictions_and_backward_is_finite(self):
        for variant in ("B", "D"):
            with self.subTest(variant=variant):
                model = build_model(self.wave, self.means, self.variances,
                                    variant=variant, patch_size=7).eval()
                inputs = torch.randn(2, 32, 7, 7)
                mask = torch.ones(2, 32, dtype=torch.bool)
                mask[:, 7:] = False
                first = model(inputs, mask)["logits"]
                changed = inputs.clone()
                changed[:, 7:] = 1000
                second = model(changed, mask)["logits"]
                self.assertEqual(tuple(first.shape), (2, 3))
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                torch.nn.functional.cross_entropy(first, torch.tensor([0, 1])).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))

    def test_invalid_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            build_model(self.wave[::-1], self.means, self.variances)


if __name__ == "__main__":
    unittest.main()

