"""Per-vehicle final analysis on the best snapshot(s): plate + helmet, saved to results/."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2

import config


def make_results_dir(root: str = "results") -> Path:
    d = Path(root) / datetime.now().strftime("run_%Y%m%d_%H%M%S") / "vehicles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _vehicle_box_in_crop(rec: dict):
    ox, oy = rec["origin"]
    x1, y1, x2, y2 = rec["bbox"]
    return (x1 - ox, y1 - oy, x2 - ox, y2 - oy)


def aggregate_helmet(per_snap: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine head analyses from up to 3 snapshots of the SAME vehicle."""
    riders = [s["heads"][0] for s in per_snap if s["heads"]]
    pillions = [s["heads"][1] for s in per_snap if len(s["heads"]) > 1]
    if not riders:
        return {"rider": None, "pillion": None}

    def combine(items):
        wsum = sum(i["conf"] for i in items) or 1.0
        p_no = sum(i["p_no"] * i["conf"] for i in items) / wsum
        return {"p_no": p_no, "conf": max(i["conf"] for i in items), "n_obs": len(items),
                "status": "no_helmet" if p_no >= 0.5 else "helmet"}

    need = 1 if len(per_snap) == 1 else 2          # pillion must be seen in >=2 snapshots (or 1 if only one)
    pill = combine(pillions) if (len(pillions) >= need or any(p["conf"] >= 0.6 for p in pillions)) else None
    return {"rider": combine(riders), "pillion": pill}


def _label(h) -> str:
    if h is None:
        return "None"
    word = "NO HELMET" if h["status"] == "no_helmet" else "HELMET"
    return f"{word} ({h['conf']:.2f})"


def _verdict(r, p) -> str:
    if r is None:
        return "UNVERIFIED (no head detected)"
    bad_r, bad_p = r["status"] == "no_helmet", bool(p and p["status"] == "no_helmet")
    if bad_r and bad_p:
        return "VIOLATION (Rider + Pillion No Helmet)"
    if bad_r:
        return "VIOLATION (Rider No Helmet)"
    if bad_p:
        return "VIOLATION (Pillion No Helmet)"
    weak = [h for h in (r, p) if h and (h["conf"] < 0.45 or 0.35 < h["p_no"] < 0.5)]
    if weak:
        return "REVIEW (low confidence)"
    return "COMPLIANT"


def analyze_vehicle(tid: int, vehicle_class: str, rec: dict, ocr, helmet, out_dir: Path) -> Dict[str, Any]:
    img = rec["crop"]
    alts = rec.get("alts", [])
    tw = config.is_two_wheeler(vehicle_class)
    row: Dict[str, Any] = {
        "track_id": tid, "vehicle_class": vehicle_class, "plate": "not detected", "plate_status": "none",
        "plate_conf": "", "rider_helmet": "N/A", "pillion_helmet": "N/A", "n_people": 0, "verdict": "N/A",
        "snapshot": "", "snapshot_frame": rec["frame_idx"],
    }
    vis = img.copy()

    # ---- plate: best snapshot first, alternates as backup ----
    plate_box = None
    if ocr is not None and vehicle_class in ("Car", "Truck", "Bus", "Auto Rickshaw", "Bike/Motorcycle", "Van"):
        try:
            r = ocr.read_snapshot_plate(tid, img, two_wheeler=tw or vehicle_class == "Auto Rickshaw",
                                        extra_imgs=[a["crop"] for a in alts])
        except Exception as exc:
            print(f"[Analyzer] plate error track {tid}: {exc}")
            r = {"status": "none", "plate": None, "conf": 0.0, "box": None}
        plate_box = r.get("box")
        if r["status"] == "valid" and r["plate"]:
            row.update(plate=r["plate"], plate_status="valid", plate_conf=f"{r['conf'] * 100:.1f}%")
        elif ocr.get_plate(tid):
            row.update(plate=ocr.get_plate(tid), plate_status="live",
                       plate_conf=f"{ocr.plate_confidences.get(tid, 0) * 100:.1f}%")
        elif r["status"] == "partial" and r["plate"]:
            row.update(plate=r["plate"] + "?", plate_status="partial", plate_conf=f"{r['conf'] * 100:.1f}%")
    if plate_box:
        cv2.rectangle(vis, tuple(plate_box[:2]), tuple(plate_box[2:]), (255, 200, 0), 3)
        cv2.putText(vis, row["plate"], (plate_box[0], max(18, plate_box[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

    # ---- helmet: two-wheelers ONLY, vote over up to 3 snapshots ----
    if tw and helmet is not None:
        per_snap = []
        for s in [rec] + alts:
            try:
                per_snap.append(helmet.analyze_snapshot(s["crop"], _vehicle_box_in_crop(s)))
            except Exception as exc:
                print(f"[Analyzer] helmet error track {tid}: {exc}")
        agg = aggregate_helmet(per_snap)
        r_h, p_h = agg["rider"], agg["pillion"]
        row["n_people"] = (1 if r_h else 0) + (1 if p_h else 0)
        row["rider_helmet"] = _label(r_h) if r_h else "NOT VISIBLE"
        row["pillion_helmet"] = _label(p_h)
        row["verdict"] = _verdict(r_h, p_h)
        for d in (per_snap[0]["heads"] if per_snap else []):
            x1, y1, x2, y2 = map(int, d["box"])
            col = (0, 0, 255) if d["status"] == "no_helmet" else (0, 200, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
            cv2.putText(vis, f"{d['status']} {d['conf']:.2f}", (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    tag = row["plate"].replace("?", "_partial") if row["plate"] != "not detected" else "noplate"
    fname = f"id{tid:04d}_{vehicle_class.replace('/', '-').replace(' ', '')}_{tag}.jpg"
    cv2.imwrite(str(out_dir / fname), vis)
    cv2.imwrite(str(out_dir / ("raw_" + fname)), img)
    row["snapshot"] = str(out_dir / fname)
    return row


def write_vehicle_csv(rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir.parent / "vehicle_report.csv"
    cols = ["track_id", "vehicle_class", "plate", "plate_status", "plate_conf", "rider_helmet",
            "pillion_helmet", "n_people", "verdict", "snapshot_frame", "snapshot"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["track_id"]):
            w.writerow(r)
    return path