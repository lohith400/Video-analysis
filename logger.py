"""
logger.py — Logs all traffic detections to CSV files
"""

import csv
import os
from datetime import datetime


class TrafficLogger:
    def __init__(self, output_dir: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── Vehicle log ───────────────────────────────────────────────────────
        self.vehicle_path = os.path.join(output_dir, f"vehicles_{ts}.csv")
        self.vehicle_file = open(self.vehicle_path, "w", newline="")
        self.vehicle_writer = csv.DictWriter(self.vehicle_file, fieldnames=[
            "timestamp", "track_id", "vehicle_type", "plate_number",
            "position_x", "position_y"
        ])
        self.vehicle_writer.writeheader()

        # ── Person log ────────────────────────────────────────────────────────
        self.person_path = os.path.join(output_dir, f"persons_{ts}.csv")
        self.person_file = open(self.person_path, "w", newline="")
        self.person_writer = csv.DictWriter(self.person_file, fieldnames=[
            "timestamp", "track_id", "helmet_status", "gender",
            "estimated_age", "age_group"
        ])
        self.person_writer.writeheader()

        print(f"  📝 Logging to:\n     {self.vehicle_path}\n     {self.person_path}")

    def log_vehicle(self, track_id, vehicle_type, plate_number,
                    pos_x=None, pos_y=None):
        self.vehicle_writer.writerow({
            "timestamp":    datetime.now().isoformat(),
            "track_id":     track_id,
            "vehicle_type": vehicle_type,
            "plate_number": plate_number or "",
            "position_x":   pos_x or "",
            "position_y":   pos_y or "",
        })
        self.vehicle_file.flush()

    def log_person(self, track_id, helmet_status, gender, age, age_group):
        self.person_writer.writerow({
            "timestamp":     datetime.now().isoformat(),
            "track_id":      track_id,
            "helmet_status": helmet_status,
            "gender":        gender,
            "estimated_age": age or "",
            "age_group":     age_group,
        })
        self.person_file.flush()

    def close(self):
        self.vehicle_file.close()
        self.person_file.close()
        print(f"\n📁 Logs saved!")
        print(f"   Vehicles → {self.vehicle_path}")
        print(f"   Persons  → {self.person_path}")
