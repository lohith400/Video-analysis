"""Unit tests for helmet_logic.classify_riders (rider/pillion positional heuristic)."""

from __future__ import annotations

import unittest

from helmet_logic import classify_riders


def person(x_center: float, class_name: str, bbox=(0, 0, 10, 10), conf=0.9):
    return {"bbox": bbox, "x_center": x_center, "class_name": class_name, "conf": conf}


class TestClassifyRiders(unittest.TestCase):
    def test_no_detections_returns_unknown_status(self):
        violations, status = classify_riders([], track_id=1, plate="KA05MH1234",
                                               vehicle_class="motorcycle", timestamp="t0")
        self.assertEqual(violations, [])
        self.assertEqual(status["rider_helmet"], "unknown")
        self.assertEqual(status["pillion_helmet"], "none")

    def test_single_rider_with_helmet_no_violation(self):
        detected = [person(50, "with_helmet")]
        violations, status = classify_riders(detected, 1, "KA05MH1234", "motorcycle", "t0")
        self.assertEqual(violations, [])
        self.assertEqual(status["rider_helmet"], "helmet")

    def test_single_rider_without_helmet_flags_violation(self):
        detected = [person(50, "without_helmet")]
        violations, status = classify_riders(detected, 1, "KA05MH1234", "motorcycle", "t0")
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["violation_type"], "rider_no_helmet")
        self.assertEqual(status["rider_helmet"], "no_helmet")

    def test_leftmost_person_is_always_rider_regardless_of_list_order(self):
        """Core positional-heuristic behavior: sorting must happen inside
        classify_riders, not depend on caller pre-sorting the list."""
        # Pillion (x=80) appears first in the input list, rider (x=20) second.
        detected = [person(80, "with_helmet"), person(20, "without_helmet")]
        violations, status = classify_riders(detected, 1, "KA05MH1234", "motorcycle", "t0")
        self.assertEqual(status["rider_helmet"], "no_helmet")     # leftmost (x=20)
        self.assertEqual(status["pillion_helmet"], "helmet")      # rightmost (x=80)
        self.assertEqual(violations[0]["violation_type"], "rider_no_helmet")

    def test_two_riders_both_without_helmet_flags_two_violations(self):
        detected = [person(20, "without_helmet"), person(80, "without_helmet")]
        violations, status = classify_riders(detected, 1, "KA05MH1234", "motorcycle", "t0")
        self.assertEqual(len(violations), 2)
        types = {v["violation_type"] for v in violations}
        self.assertEqual(types, {"rider_no_helmet", "pillion_no_helmet"})

    def test_bare_head_class_counts_as_no_helmet(self):
        """config.HELMET_CLASS_MAP maps 'head' (bare head detection) to no_helmet."""
        detected = [person(50, "head")]
        violations, status = classify_riders(detected, 1, "KA05MH1234", "motorcycle", "t0")
        self.assertEqual(status["rider_helmet"], "no_helmet")

    def test_third_and_later_detections_ignored(self):
        """Only the two leftmost detections are assigned a role — extra
        detections (e.g. a nearby pedestrian) must not affect the result."""
        detected = [
            person(10, "with_helmet"),
            person(50, "with_helmet"),
            person(90, "without_helmet"),  # ignored — not rider or pillion
        ]
        violations, status = classify_riders(detected, 1, "KA05MH1234", "motorcycle", "t0")
        self.assertEqual(violations, [])
        self.assertEqual(status["rider_helmet"], "helmet")
        self.assertEqual(status["pillion_helmet"], "helmet")

    def test_violation_bbox_is_offset_by_vehicle_bbox(self):
        """person_bbox in a violation record must be in full-frame coords,
        i.e. vehicle_bbox offset + person bbox (person bbox is relative to
        the vehicle crop)."""
        detected = [person(50, "without_helmet", bbox=(5, 5, 15, 15))]
        violations, _ = classify_riders(
            detected, 1, "KA05MH1234", "motorcycle", "t0", vehicle_bbox=(100, 200, 300, 400)
        )
        self.assertEqual(violations[0]["person_bbox"], (105, 205, 115, 215))


if __name__ == "__main__":
    unittest.main()
