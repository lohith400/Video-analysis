"""
person_analyzer.py — Gender & Age estimation using DeepFace
Falls back gracefully if no face is detected.
"""

import cv2
import numpy as np

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("  ⚠️  DeepFace not installed — gender/age analysis disabled")
    print("       Run: pip install deepface")


def classify_age_group(age: int) -> str:
    if age is None:
        return "Unknown"
    if age < 5:
        return "Toddler"
    elif age < 13:
        return "Child"
    elif age < 18:
        return "Teen"
    elif age < 60:
        return "Adult"
    else:
        return "Senior"


class PersonAnalyzer:
    def __init__(self):
        self.ready = DEEPFACE_AVAILABLE

    def analyze(self, person_crop) -> tuple[str, int | None, str]:
        """
        Returns: (gender, age, age_group)
        gender: 'Man' | 'Woman' | 'Unknown'
        age:    int or None
        age_group: 'Child' | 'Teen' | 'Adult' | 'Senior' | 'Unknown'
        """
        if not self.ready or person_crop is None or person_crop.size == 0:
            return "Unknown", None, "Unknown"

        # Focus on upper body where face is more likely
        h = person_crop.shape[0]
        upper = person_crop[:int(h * 0.6), :]
        if upper.size == 0:
            return "Unknown", None, "Unknown"

        try:
            results = DeepFace.analyze(
                upper,
                actions=["gender", "age"],
                enforce_detection=False,
                silent=True
            )
            if isinstance(results, list):
                result = results[0]
            else:
                result = results

            gender    = result.get("dominant_gender", "Unknown")
            age       = int(result.get("age", 0)) or None
            age_group = classify_age_group(age)

            # Normalize gender label
            if gender.lower() in ["man", "male"]:
                gender = "Male"
            elif gender.lower() in ["woman", "female"]:
                gender = "Female"

            return gender, age, age_group

        except Exception:
            return "Unknown", None, "Unknown"
