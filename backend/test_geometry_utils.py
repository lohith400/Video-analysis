"""Unit tests for geometry_utils: IoU and the child-height heuristic."""

from __future__ import annotations

import unittest

from geometry_utils import calculate_iou, is_child_by_height


class TestCalculateIoU(unittest.TestCase):
    def test_identical_boxes_iou_is_one(self):
        box = (0, 0, 10, 10)
        self.assertAlmostEqual(calculate_iou(box, box), 1.0)

    def test_non_overlapping_boxes_iou_is_zero(self):
        self.assertEqual(calculate_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_partial_overlap(self):
        # Two 10x10 boxes overlapping in a 5x5 region -> IoU = 25 / (100+100-25) = 25/175
        iou = calculate_iou((0, 0, 10, 10), (5, 5, 15, 15))
        self.assertAlmostEqual(iou, 25 / 175, places=4)

    def test_zero_area_box_does_not_crash(self):
        self.assertEqual(calculate_iou((0, 0, 0, 0), (0, 0, 10, 10)), 0.0)


class TestIsChildByHeight(unittest.TestCase):
    def test_below_threshold_is_child(self):
        # height is 50% of avg adult height, ratio is 0.60 -> below threshold -> child
        self.assertTrue(is_child_by_height(height=50, avg_adult_height=100, ratio=0.60))

    def test_above_threshold_not_child(self):
        self.assertFalse(is_child_by_height(height=90, avg_adult_height=100, ratio=0.60))

    def test_exactly_at_threshold_not_child(self):
        """Strictly less-than: exactly at the ratio boundary should NOT be flagged."""
        self.assertFalse(is_child_by_height(height=60, avg_adult_height=100, ratio=0.60))

    def test_no_avg_adult_height_yet_never_flags_child(self):
        """With no adult reference height in the scene yet (avg=0), the
        heuristic must not fire — this is the guard against false
        positives in low-population or startup scenes."""
        self.assertFalse(is_child_by_height(height=30, avg_adult_height=0.0, ratio=0.60))


if __name__ == "__main__":
    unittest.main()
