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
]


class CSVLogger:
    def __init__(
        self,
        counts_getter: Callable[[], Dict[str, int]],
        plates_getter: Callable[[], Dict[int, str]],
    ):
        self._counts_getter = counts_getter
        self._plates_getter = plates_getter
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
