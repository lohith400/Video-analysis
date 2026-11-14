import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plate_fusion import Read, fuse, best_template
from helmet_gate import gate, filter_persons


def test_voting_beats_single_bad_reads():
    reads = [Read("KAO5LN6085", .6, 1)] * 4 + [Read("KA05LM6085", .5, 1)] + [Read("2KA05LN6085", .5, 1)]
    f = fuse(reads)
    assert f.plate == "KA05LN6085" and f.status == "valid"


def test_two_letter_series_not_outvoted_by_truncated_reads():
    f = fuse([Read("KA05HY2091", .6, 1)] * 3 + [Read("KA05H2091", .6, 1)] * 2)
    assert f.plate == "KA05HY2091"


def test_class_constrained_positions():
    assert best_template("KAO5NF9Z74")[0] == "KA05NF9274"


def test_systematic_confusion_is_flagged_not_valid():
    f = fuse([Read("KA05NH8088", .7, 1)] * 5)
    assert f.status == "review" and "KA05NM8088" in f.alternatives


def test_no_plate_returns_none():
    assert fuse([Read("XYZ", .9, 1)]).plate is None


def test_gate_rejects_wheel_edge_and_background_person():
    rider = (150, 415, 778, 1451)
    bg = (68, 500, 120, 640)
    heads = [{"box": (448, 371, 665, 599), "conf": .17},        # real head
             {"box": (107, 1285, 315, 1523), "conf": .22},      # wheel
             {"box": (506, 1869, 534, 1901), "conf": .54},      # bottom edge
             {"box": (68, 515, 111, 571), "conf": .35}]         # pedestrian
    kept = gate(heads, 985, 1905, [rider, bg], None)
    assert [k["box"] for k in kept] == [(448, 371, 665, 599)]


def test_gate_phantom_pillion_on_seat_rejected():
    rider = (292, 412, 872, 1594)
    kept = gate([{"box": (474, 436, 711, 692), "conf": .91}, {"box": (573, 1007, 803, 1259), "conf": .66}],
                1167, 2087, [rider], None)
    assert len(kept) == 1 and kept[0]["conf"] == .91


def test_top_clipped_head_is_flagged():
    kept = gate([{"box": (644, 0, 765, 71), "conf": .71}], 1355, 1386, [(500, 0, 900, 900)], None)
    assert kept and kept[0]["clipped_top"]
