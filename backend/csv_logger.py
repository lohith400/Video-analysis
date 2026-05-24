"""Background CSV logger — appends one row per second, never overwrites."""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from typing import Callable, Dict

import config

CSV_HEADER = [
    "timestamp",
    "total_vehicles",
    "cars",
    "trucks",
    "buses",
    "auto_rickshaws",
    "motorcycles",
    "scooters",
    "bicycles",
    "plates_detected",
    "helmet_violations",
    "violation_details",
    "pedestrians_detected",
    "males",
    "females",
    "children",
]


class CSVLogger:
    def __init__(
        self,
        counts_getter: Callable[[], Dict[str, int]],
        plates_getter: Callable[[], Dict[int, str]],
        violations_getter: Callable[[], List[Dict]] = None,
        pedestrians_getter: Callable[[], Dict] = None,
    ):
        self._counts_getter = counts_getter
        self._plates_getter = plates_getter
        self._violations_getter = violations_getter if violations_getter else (lambda: [])
        self._pedestrians_getter = pedestrians_getter if pedestrians_getter else (lambda: {"total": 0, "males": 0, "females": 0, "children": 0})
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ensure_header()

    def _ensure_header(self) -> None:
        try:
            with open(config.CSV_PATH, "r", encoding="utf-8") as f:
                first = f.readline().strip()
                if first and first.split(",")[0] == "timestamp":
                    return
        except FileNotFoundError:
            pass

        with open(config.CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while self._running:
            time.sleep(config.CSV_UPDATE_INTERVAL)
            if not self._running:
                break
            self._append_row()

    def _append_row(self) -> None:
        counts = self._counts_getter()
        plates = self._plates_getter()
        plates_str = self._format_plates(plates)
        
        violations = self._violations_getter()
        pedestrians = self._pedestrians_getter()

        helmet_violations = len(violations)
        violation_details = "none"
        if violations:
            violation_details = "|".join([f"{v['track_id']}:{v['plate']}:{v['violation_type']}" for v in violations])

        pedestrians_detected = pedestrians.get("total", 0)
        males = pedestrians.get("males", 0)
        females = pedestrians.get("females", 0)
        children = pedestrians.get("children", 0)

        # Handle new TrafficCounter counts (capitalized mapped keys)
        if "Car" in counts or "Bike/Motorcycle" in counts:
            row = [
                datetime.now().isoformat(timespec="seconds"),
                counts.get("total", 0),
                counts.get("Car", 0),
                counts.get("Truck", 0),
                counts.get("Bus", 0),
                counts.get("Auto Rickshaw", 0),
                counts.get("Bike/Motorcycle", 0),  # Merged motorcycles and scooters
                0,                                 # Set scooters to 0 since they are merged
                counts.get("Bicycle", 0),
                plates_str,
                helmet_violations,
                violation_details,
                pedestrians_detected,
                males,
                females,
                children,
            ]
        else:
            # Fallback to legacy raw counts
            row = [
                datetime.now().isoformat(timespec="seconds"),
                counts.get("total", 0),
                counts.get("car", 0),
                counts.get("truck", 0),
                counts.get("bus", 0),
                counts.get("auto-rickshaw", 0),
                counts.get("motorcycle", 0),
                counts.get("scooter", 0),
                counts.get("bicycle", 0),
                plates_str,
                helmet_violations,
                violation_details,
                pedestrians_detected,
                males,
                females,
                children,
            ]
            
        with self._lock:
            with open(config.CSV_PATH, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    @staticmethod
    def _format_plates(plates: Dict[int, str]) -> str:
        if not plates:
            return "none"
        parts = [f"{tid}:{text}" for tid, text in sorted(plates.items())]
        return "|".join(parts)
