"""Unit tests for TrafficCounter class (line intersection and class voting)."""

from __future__ import annotations

import unittest
from traffic_counter import TrafficCounter
from tracker import TrackedVehicle


class TestTrafficCounter(unittest.TestCase):
    def setUp(self):
        # Initialize a basic counter with horizontal line at 50% of 100px height
        self.counter = TrafficCounter(
            line_pct=(0.0, 0.5, 1.0, 0.5),
            class_mapping={
                "car": "Car",
                "motorcycle": "Bike/Motorcycle",
                "scooter": "Bike/Motorcycle",
                "bus": "Bus",
            },
            min_track_age=2,
        )
        # Manually set frame shape as 100x100 so the line resolves to (0, 50) -> (100, 50)
        self.counter._resolve_line_coords(100, 100)

    def test_line_intersection_positive(self):
        """Verify that a path crossing the line from above to below returns True."""
        A = (0, 50)    # Line start
        B = (100, 50)  # Line end
        C = (50, 40)   # Vehicle prev bottom-center (above line)
        D = (50, 60)   # Vehicle curr bottom-center (below line)
        
        self.assertTrue(TrafficCounter.check_intersection(A, B, C, D))

    def test_line_intersection_negative(self):
        """Verify that a path that doesn't reach the line returns False."""
        A = (0, 50)
        B = (100, 50)
        C = (50, 10)
        D = (50, 45)
        
        self.assertFalse(TrafficCounter.check_intersection(A, B, C, D))

    def test_line_intersection_parallel(self):
        """Verify that moving parallel to the line returns False."""
        A = (0, 50)
        B = (100, 50)
        C = (10, 40)
        D = (90, 40)
        
        self.assertFalse(TrafficCounter.check_intersection(A, B, C, D))

    def test_class_voting(self):
        """Verify class smoothing majority vote filters out noisy single-frame predictions."""
        # Setup class history: 1 'bus', 3 'car'
        self.counter.track_classes[42] = ["car", "bus", "car", "car"]
        voted = self.counter.get_voted_class(42)
        
        self.assertEqual(voted, "Car")

    def test_track_age_filter_and_double_counting(self):
        """Verify vehicles must reach min track age before crossing is checked, and are counted exactly once."""
        # 1. Update with age 1 (below min_track_age=2)
        v1 = TrackedVehicle(track_id=1, bbox=(40, 20, 60, 40), confidence=0.9, vehicle_class="car")
        self.counter.update([v1], (100, 100))
        
        # Move vehicle so it crosses line (y2 goes from 40 to 60)
        # v2 still has age 2 (since it's the second update for track_id=1)
        v2 = TrackedVehicle(track_id=1, bbox=(40, 40, 60, 60), confidence=0.9, vehicle_class="car")
        self.counter.update([v2], (100, 100))
        
        # Verify vehicle was counted
        counts = self.counter.get_counts()
        self.assertEqual(counts["Car"], 1)
        self.assertEqual(counts["total"], 1)

        # 2. Try to cross the line again (should not double count)
        v3 = TrackedVehicle(track_id=1, bbox=(40, 30, 60, 45), confidence=0.9, vehicle_class="car")
        self.counter.update([v3], (100, 100))
        v4 = TrackedVehicle(track_id=1, bbox=(40, 45, 60, 65), confidence=0.9, vehicle_class="car")
        self.counter.update([v4], (100, 100))

        counts = self.counter.get_counts()
        self.assertEqual(counts["Car"], 1)  # Stays at 1!
        self.assertEqual(counts["total"], 1)


if __name__ == "__main__":
    unittest.main()
