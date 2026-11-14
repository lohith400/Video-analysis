"""Multi-frame plate fusion: turns many noisy OCR reads of ONE vehicle into one reliable plate.

Why this exists
---------------
A single OCR read of a small, slightly blurred plate is wrong by one character
surprisingly often (H<->M, Y<->I, 1<->4, D<->0 ...). The same confusion tends to repeat in
neighbouring frames, so "two reads agree" does not mean "correct". What does work is looking
at MANY frames of the same vehicle (the plate grows as it approaches the camera) and voting
character by character, with every position restricted to the character class that Indian
plates allow there:

    SS DD L{1,3} DDDD      e.g. KA 05 HY 2091   (state, RTO, series letters, number)

Nothing here needs a GPU or a model; it is plain, unit-testable logic.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from plate_utils import VALID_STATES

PRIOR_STATE = "KA"

# What OCR typically confuses. LETTER_AS_DIGIT: a letter seen where a digit must be.
LETTER_AS_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "Z": "2", "S": "5",
    "G": "6", "B": "8", "A": "4", "T": "7", "J": "3",
}
DIGIT_AS_LETTER = {
    "0": "O", "1": "I", "2": "Z", "3": "J", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B",
}
# Letters that look like their neighbours; used only to *label* ambiguity, never to rewrite.
LOOKALIKES = {
    "H": "MNKA", "M": "HNW", "N": "MHK", "Y": "VIT", "I": "LJ1Y", "D": "O0Q", "O": "D0Q",
    "8": "B3", "B": "8", "5": "S6", "6": "5G", "1": "I7", "4": "A", "0": "OD",
}

# Pairs this OCR stack gets wrong systematically (measured on L3: all three M->H slips, Y->I,
# 1->7). Voting cannot fix a bias that repeats in every frame, so a plate containing one of these
# is never reported as plain "valid": it is flagged for review with the swapped alternative.
SWAP = {"H": "M", "M": "H", "Y": "I", "I": "Y", "1": "7", "7": "1"}

# Indian series letters never use I or O (MoRTH rule); seeing one means the glyph is another letter.
INVALID_SERIES = {"I", "O"}
SERIES_ALTS = ["Y", "L", "T", "J"]

SERIES_LENS = (1, 2, 3)          # letters after the RTO digits
TOTAL_LENS = tuple(4 + n + 4 for n in SERIES_LENS)   # 9, 10, 11


@dataclass
class Read:
    """One OCR hypothesis from one frame/variant."""
    text: str
    conf: float = 0.5          # OCR confidence 0..1
    weight: float = 1.0        # frame quality (plate size, sharpness, detector conf)
    src: str = ""


@dataclass
class FusedPlate:
    plate: Optional[str]
    status: str                                   # valid | review | none
    conf: float = 0.0                             # 0..1 mean per-character agreement
    n_reads: int = 0
    n_used: int = 0
    weak_positions: List[int] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    per_char: List[Dict[str, float]] = field(default_factory=list)
    ambiguous_positions: List[int] = field(default_factory=list)


def clean(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def _as_letter(ch: str) -> Tuple[str, float]:
    if ch.isalpha():
        return ch, 0.0
    return DIGIT_AS_LETTER.get(ch, ""), 1.0


def _as_digit(ch: str) -> Tuple[str, float]:
    if ch.isdigit():
        return ch, 0.0
    return LETTER_AS_DIGIT.get(ch, ""), 1.0


def best_template(text: str) -> Optional[Tuple[str, float]]:
    """Fit a noisy string to the Indian plate template. Returns (candidate, cost) or None.

    Cost = character substitutions forced by the template + skipped (junk) characters
    + a penalty when the state code had to be replaced by the prior.
    A window may start/end inside the string so stray border characters are ignored.
    """
    s = clean(text)
    best: Optional[Tuple[str, float]] = None
    for n_series in SERIES_LENS:
        total = 4 + n_series + 4
        if len(s) < total:
            continue
        for start in range(0, len(s) - total + 1):
            w = s[start:start + total]
            # junk before the plate is cheap (frame/logo glyphs); dropping the TAIL is not: plates end
            # in four digits and a lost last digit is far likelier an OCR miss than border junk
            cost = 0.35 * start + 1.3 * (len(s) - start - total)
            out: List[str] = []
            ok = True
            for i, ch in enumerate(w):
                if i < 2:                                # state letters
                    c, k = _as_letter(ch)
                elif i < 4:                              # RTO digits
                    c, k = _as_digit(ch)
                elif i < 4 + n_series:                   # series letters
                    c, k = _as_letter(ch)
                else:                                    # 4 digits
                    c, k = _as_digit(ch)
                if not c:
                    ok = False
                    break
                cost += k
                out.append(c)
            if not ok:
                continue
            cand = "".join(out)
            if cand[:2] not in VALID_STATES:
                if cand[0] == "K" or cand[1] == "A":     # damaged 'KA' prefix
                    cand = PRIOR_STATE + cand[2:]
                    cost += 1.0
                else:
                    continue
            if best is None or cost < best[1]:
                best = (cand, cost)
    return best


def _candidates(reads: Iterable[Read], max_cost: float) -> List[Tuple[str, float]]:
    """(candidate_plate, weight) for every read that fits the template."""
    out = []
    for r in reads:
        fit = best_template(r.text)
        if fit is None or fit[1] > max_cost:
            continue
        cand, cost = fit
        # confident, cheap-to-fit reads count more; heavily repaired reads count little
        w = max(r.weight, 1e-3) * (0.25 + r.conf) * (0.55 ** cost)
        out.append((cand, w))
    return out


def fuse(reads: Sequence[Read], max_cost: float = 2.6, min_reads: int = 1) -> FusedPlate:
    """Vote character by character over all template-fitting reads."""
    cands = _candidates(reads, max_cost)
    if len(cands) < min_reads or not cands:
        return FusedPlate(None, "none", n_reads=len(reads))

    # 1) agree on the plate STRUCTURE (length) first: a 2-letter series must not be outvoted
    #    by 1-letter reads that merely lost a character.
    by_len: Dict[int, float] = defaultdict(float)
    for c, w in cands:
        by_len[len(c)] += w
    length = max(by_len, key=by_len.get)
    items = [(c, w) for c, w in cands if len(c) == length]

    # 2) per-position weighted vote
    chars: List[str] = []
    per_char: List[Dict[str, float]] = []
    weak: List[int] = []
    shares: List[float] = []
    for i in range(length):
        tally: Dict[str, float] = defaultdict(float)
        for c, w in items:
            tally[c[i]] += w
        tot = sum(tally.values()) or 1.0
        ranked = sorted(tally.items(), key=lambda kv: kv[1], reverse=True)
        chars.append(ranked[0][0])
        shares.append(ranked[0][1] / tot)
        per_char.append({k: round(v / tot, 3) for k, v in ranked[:3]})
        runner = ranked[1][1] / tot if len(ranked) > 1 else 0.0
        if ranked[0][1] / tot < 0.70 or runner > 0.30:
            weak.append(i)
    plate = "".join(chars)

    # 3) alternatives: the weak positions flipped to their runner-up (for human review)
    alts: List[str] = []
    for i in weak[:3]:
        if len(per_char[i]) > 1:
            runner = list(per_char[i])[1]
            alts.append(plate[:i] + runner + plate[i + 1:])

    conf = sum(shares) / len(shares)
    n_used = len(items)
    n_series = len(plate) - 8
    ambig = [i for i, ch in enumerate(plate)
             if ch in SWAP and not (i < 2) and (ch not in "17" or i in (2, 3))]
    bad_series = [i for i in range(4, 4 + n_series) if plate[i] in INVALID_SERIES]
    for i in bad_series:
        if i not in ambig:
            ambig.append(i)
        for a in SERIES_ALTS:
            alts.append(plate[:i] + a + plate[i + 1:])
    for i in ambig:
        if i in bad_series:
            continue
        alts.append(plate[:i] + SWAP[plate[i]] + plate[i + 1:])
    alts = list(dict.fromkeys(a for a in alts if a != plate))[:5]
    status = "valid" if (not weak and not ambig and n_used >= 2) else "review"
    return FusedPlate(plate, status, round(conf, 3), len(reads), n_used, weak, alts, per_char, ambig)
