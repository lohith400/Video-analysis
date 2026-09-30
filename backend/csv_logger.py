import os
import csv
import time
import threading
from datetime import datetime
from typing import Callable, Any, Dict, List, Optional


class CSVLogger:
    def __init__(
        self,
        output_path: str = "traffic_log.csv",
        interval: float = 1.0,
        counts_getter: Optional[Callable[[], Dict[str, Any]]] = None,
        plates_getter: Optional[Callable[[], Any]] = None,
        violations_getter: Optional[Callable[[], Any]] = None,
        pedestrians_getter: Optional[Callable[[], Any]] = None,
        **kwargs
    ):
        self.output_path = output_path
        self.interval = interval
        self.counts_getter = counts_getter
        self.plates_getter = plates_getter
        self.violations_getter = violations_getter
        self.pedestrians_getter = pedestrians_getter

        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.fieldnames = [
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
            "children"
        ]

        self._init_csv()

    def _init_csv(self):
        with self.lock:
            target_dir = os.path.dirname(os.path.abspath(self.output_path))
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            with open(self.output_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def _format_plates(self, raw_plates: Any) -> str:
        if not raw_plates:
            return "none"
        if isinstance(raw_plates, dict):
            items = [f"{k}:{v}" for k, v in raw_plates.items() if v and v != "not detected"]
            return "|".join(items) if items else "none"
        if isinstance(raw_plates, list):
            items = []
            for p in raw_plates:
                if isinstance(p, dict):
                    tid = p.get("track_id", p.get("id", ""))
                    txt = p.get("plate", p.get("text", ""))
                    if txt and txt != "not detected":
                        items.append(f"{tid}:{txt}" if tid else str(txt))
                elif p and p != "not detected":
                    items.append(str(p))
            return "|".join(items) if items else "none"
        return str(raw_plates)

    def _format_violations(self, raw_violations: Any) -> tuple[int, str]:
        if not raw_violations:
            return 0, "none"
        items = []
        if isinstance(raw_violations, list):
            for v in raw_violations:
                if isinstance(v, dict):
                    tid = v.get("track_id", v.get("id", "?"))
                    vtype = v.get("violation", v.get("type", "no_helmet"))
                    items.append(f"{tid}:{vtype}")
                elif v and v != "none":
                    items.append(str(v))
        return len(items), ("|".join(items) if items else "none")

    def log_snapshot(self):
        counts = self.counts_getter() if self.counts_getter else {}
        plates = self.plates_getter() if self.plates_getter else None
        violations = self.violations_getter() if self.violations_getter else None
        pedestrians = self.pedestrians_getter() if self.pedestrians_getter else {}

        current_total = counts.get("total", counts.get("total_vehicles", 0))
        if current_total == 0 and not plates and not violations:
            return

        v_count, v_details = self._format_violations(violations)
        ped_count = pedestrians if isinstance(pedestrians, int) else (pedestrians.get("total", 0) if isinstance(pedestrians, dict) else 0)

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "total_vehicles": current_total,
            "cars": counts.get("Car", counts.get("cars", 0)),
            "trucks": counts.get("Truck", counts.get("trucks", 0)),
            "buses": counts.get("Bus", counts.get("buses", 0)),
            "auto_rickshaws": counts.get("Auto Rickshaw", counts.get("auto_rickshaws", 0)),
            "motorcycles": counts.get("Bike/Motorcycle", counts.get("motorcycles", counts.get("bike", 0))),
            "scooters": counts.get("Scooter", counts.get("scooters", 0)),
            "bicycles": counts.get("Bicycle", counts.get("bicycles", 0)),
            "plates_detected": self._format_plates(plates),
            "helmet_violations": v_count,
            "violation_details": v_details,
            "pedestrians_detected": ped_count,
            "males": pedestrians.get("males", 0) if isinstance(pedestrians, dict) else 0,
            "females": pedestrians.get("females", 0) if isinstance(pedestrians, dict) else 0,
            "children": pedestrians.get("children", 0) if isinstance(pedestrians, dict) else 0
        }

        with self.lock:
            with open(self.output_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row)

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self.log_snapshot()
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def write_final_report(
        self,
        vehicle_counts: Dict[str, int],
        per_vehicle_rows: Optional[List[Dict[str, Any]]] = None,
        pedestrian_totals: Optional[Dict[str, int]] = None,
        two_wheeler_rows: Optional[List[Dict[str, Any]]] = None,
        recognized_plates_rows: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs
    ):
        self.stop()
        now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        with self.lock:
            with open(self.output_path, mode="a", newline="", encoding="utf-8") as f:
                f.write("\n===== FINAL SESSION REPORT =====\n")
                f.write(f"generated_at,{now_str}\n\n")

                # Vehicle counts summary
                f.write("-- Vehicle counts by class --\n")
                f.write("vehicle_class,count\n")
                total_v = 0
                if vehicle_counts:
                    for vclass, cnt in sorted(vehicle_counts.items()):
                        if vclass.lower() in ["total", "total_counted"]:
                            continue
                        f.write(f"{vclass},{cnt}\n")
                        total_v += cnt
                f.write(f"TOTAL,{total_v}\n\n")

                # Humans / Pedestrians summary
                f.write("-- Humans detected --\n")
                f.write("category,count\n")
                if pedestrian_totals:
                    for cat, cnt in pedestrian_totals.items():
                        f.write(f"{cat},{cnt}\n")
                else:
                    f.write("total,0\nmales,0\nfemales,0\nchildren,0\nunknown,0\n")
                f.write("\n")

                # Recognized license plates detail
                if recognized_plates_rows:
                    f.write("-- Recognized license plates detail --\n")
                    f.write("track_id,vehicle_class,plate_number,confidence\n")
                    for p in recognized_plates_rows:
                        tid = p.get("track_id", "")
                        vcls = p.get("vehicle_class", "Unknown")
                        plate = p.get("plate", "")
                        conf = p.get("confidence", "")
                        f.write(f"{tid},{vcls},{plate},{conf}\n")
                    f.write("\n")

                # Two-wheeler helmet compliance detail
                if two_wheeler_rows:
                    f.write("-- Two-wheeler helmet compliance detail --\n")
                    f.write("track_id,vehicle_class,plate_number,rider_helmet,pillion_helmet,verdict\n")
                    for tw in two_wheeler_rows:
                        tid = tw.get("track_id", "")
                        vcls = tw.get("vehicle_class", "Bike/Motorcycle")
                        plate = tw.get("plate", "not detected")
                        r_h = tw.get("rider_helmet", "Unknown")
                        p_h = tw.get("pillion_helmet", "None")
                        verd = tw.get("verdict", "")
                        f.write(f"{tid},{vcls},{plate},{r_h},{p_h},{verd}\n")
                    f.write("\n")

                # Per-vehicle itemized list
                f.write("-- Per-vehicle detail (every vehicle counted this run) --\n")
                f.write("track_id,vehicle_class,plate_number,helmet_status\n")
                if per_vehicle_rows:
                    for item in per_vehicle_rows:
                        tid = item.get("track_id", "")
                        vcls = item.get("vehicle_class", "Unknown")
                        plate = item.get("plate", "not detected")
                        helmet = item.get("helmet_status", "N/A")
                        f.write(f"{tid},{vcls},{plate},{helmet}\n")