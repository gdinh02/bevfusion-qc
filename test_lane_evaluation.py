import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import types
import sys

import cv2
import numpy as np
from lane_evaluation import match_masks, metrics, rasterize_boundaries, thin_mask


class EvaluationTests(unittest.TestCase):
    def line(self, column=40):
        mask = np.zeros((100, 100), np.uint8)
        mask[20:80, column] = 255
        return mask

    def score(self, gt, pred):
        return metrics(match_masks(gt, pred)[0])

    def test_identical(self):
        self.assertEqual(self.score(self.line(), self.line()), (1, 1, 1, 1))

    def test_tolerance_at_33_ppm(self):
        self.assertEqual(self.score(self.line(), self.line(43))[:3], (1, 1, 1))
        self.assertEqual(self.score(self.line(), self.line(44))[:3], (0, 0, 0))

    def test_euclidean_not_square_distance(self):
        gt = np.zeros((20, 20), np.uint8)
        pred = gt.copy()
        gt[5, 5] = 255
        pred[8, 8] = 255
        self.assertEqual(self.score(gt, pred)[:2], (0, 0))

    def test_separate_precision_recall_counts(self):
        gt = self.line()
        gt[:, 70] = self.line()[:, 40]
        self.assertEqual(self.score(gt, self.line())[:3], (1, 0.5, 2 / 3))
        self.assertEqual(self.score(self.line(), gt)[:3], (0.5, 1, 2 / 3))

    def test_empty_masks(self):
        empty = np.zeros((100, 100), np.uint8)
        for gt, pred in ((empty, empty), (empty, self.line()), (self.line(), empty)):
            self.assertEqual(self.score(gt, pred)[:3], (0, 0, 0))

    def test_thinning(self):
        gt = self.line()
        thick = cv2.dilate(gt, np.ones((3, 3), np.uint8))
        self.assertEqual(self.score(thick, gt)[:3], (1, 1, 1))
        skeleton = thin_mask(thick > 127)
        self.assertTrue(np.array_equal(skeleton, thin_mask(skeleton)))

    def test_validation(self):
        for ppm, tol in ((0, .1), (33, -.1), (float('nan'), .1)):
            with self.assertRaises(ValueError):
                match_masks(self.line(), self.line(), ppm, tol)
        with self.assertRaises(ValueError):
            match_masks(self.line(), self.line()[:50])

    def test_renderer_ignores_styling(self):
        evaluator = types.ModuleType('lane_inference.lane_fitting')
        evaluator.evaluate_lane_polynomial = lambda c, z: np.polynomial.polynomial.polyval(z, c)
        with patch.dict(sys.modules, {'lane_inference.lane_fitting': evaluator}):
            boundary = dict(z_min=-2, z_max=2, coefficients=[0, 0, 0])
            first = rasterize_boundaries([dict(boundary, source='single_stream')], (200, 200), 33)
            second = rasterize_boundaries([dict(boundary, source='paired')], (200, 200), 33)
            self.assertTrue(np.array_equal(first, second))
            self.assertEqual(set(np.unique(first)), {0, 255})
            self.assertLessEqual((first > 0).sum(axis=1).max(), 1)

    def test_cli_and_missing_prediction(self):
        spec = importlib.util.spec_from_file_location('evaluate', Path(__file__).with_name('eval-script.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / 'frame_gt.png'), self.line())
            with self.assertRaises(FileNotFoundError):
                module.evaluate_and_visualize(directory, directory, directory)
            cv2.imwrite(str(root / 'frame_pred.png'), self.line(43))
            self.assertEqual(module.evaluate_and_visualize(directory, directory, directory)[:3], (1, 1, 1))


if __name__ == '__main__':
    unittest.main()
