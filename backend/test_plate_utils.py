"""Unit tests for plate_utils: OCR character correction, validation, NMS."""

from __future__ import annotations

import unittest

from plate_utils import (
    INDIAN_PLATE_REGEX,
    clean_and_correct_indian_plate,
    is_valid_plate,
    nms,
)


class TestPlateCleaning(unittest.TestCase):
    def test_clean_plate_passthrough(self):
        """A perfectly-read standard plate should come back unchanged."""
        result = clean_and_correct_indian_plate("KA05MH1234")
        self.assertEqual(result, "KA05MH1234")

    def test_strips_non_alphanumeric_noise(self):
        """OCR often inserts spaces/dashes — these must be stripped before validation."""
        result = clean_and_correct_indian_plate("KA-05 MH 1234")
        self.assertEqual(result, "KA05MH1234")

    def test_corrects_confused_digits_in_number_block(self):
        """Trailing digit block should have letter-lookalikes corrected to digits."""
        # 'O' -> '0', 'I' -> '1' in the last-4-digit block
        result = clean_and_correct_indian_plate("KA05MHOI34")
        self.assertEqual(result, "KA05MH0134")

    def test_corrects_common_state_code_confusions(self):
        """'MA'/'M4'/'M0' are common OCR misreads for Maharashtra's 'MH'."""
        result = clean_and_correct_indian_plate("MA05AB1234")
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("MH"))

    def test_rejects_invalid_state_code(self):
        """A state code that isn't correctable to any real Indian state must be rejected."""
        result = clean_and_correct_indian_plate("ZZ05AB1234")
        self.assertIsNone(result)

    def test_rejects_too_short(self):
        result = clean_and_correct_indian_plate("KA1")
        self.assertIsNone(result)

    def test_rejects_too_long(self):
        result = clean_and_correct_indian_plate("KA05MH123456789")
        self.assertIsNone(result)

    def test_is_valid_plate_length_bounds(self):
        self.assertTrue(is_valid_plate("KA05MH1234"))   # 10 chars, in bounds
        self.assertFalse(is_valid_plate("AB"))          # too short
        self.assertFalse(is_valid_plate("A" * 11))       # too long

    def test_indian_plate_regex_accepts_standard_format(self):
        self.assertRegex("KA05MH1234", INDIAN_PLATE_REGEX)

    def test_indian_plate_regex_rejects_malformed(self):
        self.assertNotRegex("KA5MH1234", INDIAN_PLATE_REGEX)   # missing leading zero
        self.assertNotRegex("ka05mh1234", INDIAN_PLATE_REGEX)  # lowercase not allowed


class TestNMS(unittest.TestCase):
    def test_keeps_single_box(self):
        keep = nms([(0, 0, 10, 10)], [0.9])
        self.assertEqual(keep, [0])

    def test_suppresses_heavily_overlapping_lower_score_box(self):
        boxes = [(0, 0, 10, 10), (1, 1, 11, 11)]
        scores = [0.9, 0.5]
        keep = nms(boxes, scores, iou_threshold=0.5)
        self.assertEqual(keep, [0])  # only the higher-confidence box survives

    def test_keeps_non_overlapping_boxes(self):
        boxes = [(0, 0, 10, 10), (100, 100, 110, 110)]
        scores = [0.9, 0.8]
        keep = nms(boxes, scores, iou_threshold=0.5)
        self.assertEqual(sorted(keep), [0, 1])

    def test_empty_input(self):
        self.assertEqual(nms([], []), [])


if __name__ == "__main__":
    unittest.main()
