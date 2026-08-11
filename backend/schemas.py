"""
Plain data types shared across the pipeline.

Why this file exists: TrackedVehicle used to live inside tracker.py, which
also imports `ultralytics` (and transitively torch) to do the actual YOLO
inference. That coupling meant any module that only wanted the *dataclass*
(traffic_counter.py, annotator.py, the unit tests) was forced to import
torch/ultralytics too — slow, and unnecessary for pure line-crossing math
or CI runs that don't need a GPU stack at all.

Splitting the data shape out from the model-loading logic keeps the
dependency graph honest: modules that do geometry/bookkeeping only depend
on this file, and only tracker.py itself depends on ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackedVehicle:
    track_id: int
    bbox: tuple  # x1, y1, x2, y2
    confidence: float
    vehicle_class: str
