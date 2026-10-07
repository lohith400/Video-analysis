"""
Unit tests for geometry_utils:
- IoU math
- Rider exclusion (intersection ratio & upward-expanded bounding box)
- Demographic label resolution (model priority over weak height heuristic)
- Per-track temporal voting accumulator (hysteresis, anti-flip-flop, abstention)
- Rolling median count smoothing
"""

from __future__ import annotations

import sys
from pathlib import Path
backend_dir = str(Path(__file__).resolve().parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import unittest

from geometry_utils import (
    calculate_iou,
    calculate_intersection_ratio,
    is_center_inside_expanded_box,
    is_rider_of_vehicle,
    is_child_by_height,
    resolve_demographic_label,
    TrackVoteAccumulator,
    RollingMedianSmoother,
)


class TestCalculateIoU(unittest.TestCase):
    def test_identical_boxes_iou_is_one(self):
        box = (0, 0, 10, 10)
        self.assertAlmostEqual(calculate_iou(box, box), 1.0)

    def test_non_overlapping_boxes_iou_is_zero(self):
        self.assertEqual(calculate_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_partial_overlap(self):
        iou = calculate_iou((0, 0, 10, 10), (5, 5, 15, 15))
        self.assertAlmostEqual(iou, 25 / 175, places=4)

    def test_zero_area_box_does_not_crash(self):
        self.assertEqual(calculate_iou((0, 0, 0, 0), (0, 0, 10, 10)), 0.0)


class TestRiderExclusion(unittest.TestCase):
    def test_high_intersection_ratio_flags_rider(self):
        # Person box (50, 50, 100, 150) completely inside vehicle box (40, 40, 200, 200)
        p_box = (50, 50, 100, 150)
        v_box = (40, 40, 200, 200)
        ratio = calculate_intersection_ratio(p_box, v_box)
        self.assertAlmostEqual(ratio, 1.0)
        self.assertTrue(is_rider_of_vehicle(p_box, v_box, "car"))

    def test_rider_extending_above_motorcycle_caught_by_upward_expansion(self):
        # Motorcycle box: x: 100..200, y: 150..250 (height = 100)
        # 40% upward expansion gives y: 110..250
        # Rider seated on motorcycle: x: 120..180, y: 80..180 (center is x: 150, y: 130)
        m_box = (100, 150, 200, 250)
        rider_box = (120, 80, 180, 180)
        self.assertTrue(is_center_inside_expanded_box(rider_box, m_box, upward_expansion_ratio=0.40))
        self.assertTrue(is_rider_of_vehicle(rider_box, m_box, vehicle_class="motorcycle"))

    def test_pedestrian_beside_motorcycle_not_flagged_as_rider(self):
        # Motorcycle box: x: 100..200, y: 150..250
        # Walking pedestrian to the left: x: 20..60, y: 120..220
        m_box = (100, 150, 200, 250)
        ped_box = (20, 120, 60, 220)
        self.assertFalse(is_rider_of_vehicle(ped_box, m_box, vehicle_class="motorcycle"))


class TestChildLogicPriority(unittest.TestCase):
    def test_model_child_wins_over_adult_height(self):
        # Even if person is tall (e.g., 90px against avg 100px), model child prediction wins
        lbl, conf = resolve_demographic_label(
            model_label="child",
            model_conf=0.85,
            height=90,
            avg_adult_height=100,
            use_height_heuristic=False,
            conf_threshold=0.50,
        )
        self.assertEqual(lbl, "child")
        self.assertEqual(conf, 0.85)

    def test_model_female_adult_wins_over_short_height_when_heuristic_disabled(self):
        # Short adult (50px vs avg 100px) must NOT be forced to child if model predicts female with confidence
        lbl, conf = resolve_demographic_label(
            model_label="female_adult",
            model_conf=0.75,
            height=50,
            avg_adult_height=100,
            use_height_heuristic=False,
            conf_threshold=0.50,
        )
        self.assertEqual(lbl, "female_adult")

    def test_low_confidence_abstains_to_unknown(self):
        lbl, conf = resolve_demographic_label(
            model_label="male_adult",
            model_conf=0.30,
            conf_threshold=0.50,
        )
        self.assertEqual(lbl, "unknown")


class TestTrackVoteAccumulator(unittest.TestCase):
    def test_voting_prevents_flip_flop(self):
        tracker = TrackVoteAccumulator(window_size=10, min_votes=3, weight_threshold=0.60)
        # Consistent male votes
        tracker.add_vote(track_id=1, label="male_adult", conf=0.8)
        tracker.add_vote(track_id=1, label="male_adult", conf=0.8)
        tracker.add_vote(track_id=1, label="male_adult", conf=0.8)
        lbl, _ = tracker.get_voted_label(track_id=1)
        self.assertEqual(lbl, "male_adult")

        # Single noisy female frame does not flip track label
        tracker.add_vote(track_id=1, label="female_adult", conf=0.7)
        lbl, _ = tracker.get_voted_label(track_id=1)
        self.assertEqual(lbl, "male_adult")

    def test_abstains_with_insufficient_votes(self):
        tracker = TrackVoteAccumulator(window_size=10, min_votes=3, weight_threshold=0.60)
        tracker.add_vote(track_id=2, label="female_adult", conf=0.9)
        lbl, _ = tracker.get_voted_label(track_id=2)
        self.assertEqual(lbl, "unknown")


class TestRollingMedianSmoother(unittest.TestCase):
    def test_smooths_instantaneous_spikes(self):
        smoother = RollingMedianSmoother(window_duration_sec=2.0)
        smoother.update(current_time=0.0, count=5)
        smoother.update(current_time=0.5, count=5)
        # Single-frame occlusion / false spike
        smoothed = smoother.update(current_time=1.0, count=18)
        # Median of [5, 5, 18] is 5
        self.assertEqual(smoothed, 5)

    def test_adapts_to_sustained_crowd_change(self):
        smoother = RollingMedianSmoother(window_duration_sec=2.0)
        smoother.update(current_time=0.0, count=2)
        smoother.update(current_time=1.0, count=8)
        smoother.update(current_time=1.5, count=8)
        smoothed = smoother.update(current_time=2.0, count=8)
        self.assertEqual(smoothed, 8)


if __name__ == "__main__":
    unittest.main()
