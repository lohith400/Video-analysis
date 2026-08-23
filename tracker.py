"""
tracker.py — Simple centroid-based object tracker
Assigns persistent IDs to detected objects across frames.
For production use, replace with ByteTrack or DeepSORT.
"""

import numpy as np
from collections import OrderedDict


class ObjectTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: int = 80):
        self.next_id       = 0
        self.objects       = OrderedDict()   # id → centroid
        self.labels        = OrderedDict()   # id → label
        self.disappeared   = OrderedDict()   # id → frames_missing
        self.max_disap     = max_disappeared
        self.max_distance  = max_distance

    def _centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def register(self, centroid, label):
        self.objects[self.next_id]     = centroid
        self.labels[self.next_id]      = label
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.labels[obj_id]
        del self.disappeared[obj_id]

    def update(self, detections, namespace="default"):
        """
        detections: list of (box_xyxy, label)
        Returns:    list of (track_id, box_xyxy, label)
        """
        if not detections:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disap:
                    self.deregister(obj_id)
            return []

        input_centroids = [self._centroid(d[0]) for d in detections]
        input_labels    = [d[1] for d in detections]
        input_boxes     = [d[0] for d in detections]

        if not self.objects:
            track_results = []
            for i, (c, l, b) in enumerate(
                    zip(input_centroids, input_labels, input_boxes)):
                tid = self.register(c, l)
                track_results.append((tid, b, l))
            return track_results

        # Match existing objects to new detections
        obj_ids       = list(self.objects.keys())
        obj_centroids = list(self.objects.values())

        # Distance matrix
        D = np.zeros((len(obj_centroids), len(input_centroids)))
        for i, oc in enumerate(obj_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

        # Greedy matching: sort by distance
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        matches = {}

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            obj_id = obj_ids[r]
            matches[obj_id] = c
            self.objects[obj_id]     = input_centroids[c]
            self.disappeared[obj_id] = 0
            used_rows.add(r)
            used_cols.add(c)

        # Increment disappeared for unmatched existing
        for r, obj_id in enumerate(obj_ids):
            if r not in used_rows:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disap:
                    self.deregister(obj_id)

        # Register new objects for unmatched detections
        for c in range(len(input_centroids)):
            if c not in used_cols:
                self.register(input_centroids[c], input_labels[c])

        # Build result list
        results = []
        for obj_id, col in matches.items():
            results.append((obj_id, input_boxes[col], input_labels[col]))

        return results
