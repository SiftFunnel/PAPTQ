"""CPU-only fixture checks; never loads private data or trained weights."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from public_data import IRIS_thousand  # noqa: E402
from split_data import (  # noqa: E402
    classwise_target_auxiliary_indices,
    exact_image_overlap,
    legacy_facescrub_indices,
    save_split_npz,
)


class PublicPackageTests(unittest.TestCase):
    def test_facescrub_legacy_index_split(self):
        train, test = legacy_facescrub_indices(10)
        expected = np.arange(10)
        np.random.RandomState(624).shuffle(expected)
        np.testing.assert_array_equal(train, expected[:8])
        np.testing.assert_array_equal(test, expected[8:])

    def test_target_auxiliary_isolation_and_npz_loader(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
        images = np.arange(8 * 4 * 4, dtype=np.uint8).reshape(8, 4, 4)
        split = classwise_target_auxiliary_indices(labels, range(2), 1, seed=17)
        self.assertEqual(len(split.train), 4)
        self.assertEqual(len(split.test), 2)
        self.assertEqual(len(split.auxiliary), 2)
        self.assertFalse(exact_image_overlap(images, split.train, split.test))
        with tempfile.TemporaryDirectory() as folder:
            save_split_npz(Path(folder) / "iris_thousand.npz", images, labels, split)
            self.assertEqual(len(IRIS_thousand(folder, train=True)), 4)
            self.assertEqual(len(IRIS_thousand(folder, train=False)), 2)

    def test_core_and_model_imports(self):
        import run_quantization
        from resnet50_mp import resnet50_ptq
        from transformer_quant_models import vit_ptq, swin_b_ptq
        parser = run_quantization.build_parser()
        args = parser.parse_args(["--arch", "resnet50", "--dataset", "iris_thousand"])
        self.assertEqual(args.arch, "resnet50")
        self.assertTrue(callable(run_quantization.core.run_quantization_pipeline))
        self.assertTrue(callable(resnet50_ptq))
        self.assertTrue(callable(vit_ptq))
        self.assertTrue(callable(swin_b_ptq))


if __name__ == "__main__":
    unittest.main()
