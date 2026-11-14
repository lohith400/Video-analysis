# Plate + helmet accuracy changes

New: backend/plate_evidence.py, plate_fusion.py, helmet_gate.py, tests/test_plate_helmet_fix.py
Changed: ocr_engine.py (evidence kept for whole track, read_track_plate), plate_reader.py (multi-scale reads),
vehicle_analyzer.py (fused plate, saved plate crops, new CSV columns, cut-off-head review),
helmet_checker.py (person-box head gating, conf 0.10), snapshot_manager.py (MAX_ALTS 5), main.py (headless-safe shutdown).

plate_status = valid | review | none. "review" lists plate_alternatives (checked by eye against results/run_*/plates/).
Known limit: EasyOCR reads M as H, Y as I, 1 as 7 in every frame; these are flagged review, not fixed.
Known limit: a bare head labelled "helmet" at high confidence by the model is not caught; needs retraining.
verification_run/ = output of a 120-frame L3 slice.
