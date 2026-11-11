import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Connections between the 21 MediaPipe hand landmarks.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandTracker:
    def __init__(self):
        model_path = Path(
            os.environ.get(
                "MEDIAPIPE_HAND_MODEL",
                Path(__file__).resolve().with_name(
                    "hand_landmarker.task"
                ),
            )
        )

        if not model_path.is_file():
            raise FileNotFoundError(
                "MediaPipe hand model not found. Download "
                "hand_landmarker.task and place it beside "
                "hand_tracker.py. Checked path: "
                f"{model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.detector = vision.HandLandmarker.create_from_options(
            options
        )

        self.landmarks = None
        self._last_timestamp_ms = -1

    def process(self, frame):
        """
        Detect a hand in a BGR OpenCV frame.

        Returns:
            annotated_frame: Frame with hand landmarks drawn.
            landmarks: List of 21 normalized MediaPipe landmarks,
                       or None if no hand is detected.
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # MediaPipe VIDEO mode requires increasing timestamps.
        timestamp_ms = time.monotonic_ns() // 1_000_000

        timestamp_ms = max(
            timestamp_ms,
            self._last_timestamp_ms + 1
        )

        self._last_timestamp_ms = timestamp_ms

        result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        self.landmarks = (
            result.hand_landmarks[0]
            if result.hand_landmarks
            else None
        )

        if self.landmarks is not None:
            height, width = frame.shape[:2]

            points = [
                (
                    int(lm.x * width),
                    int(lm.y * height)
                )
                for lm in self.landmarks
            ]

            # Draw hand connections.
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start_idx],
                    points[end_idx],
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            # Draw landmark points.
            for point in points:
                cv2.circle(
                    frame,
                    point,
                    4,
                    (0, 0, 255),
                    -1,
                    cv2.LINE_AA
                )

        return frame, self.landmarks

    def get_landmark(self, index):
        if self.landmarks is not None:
            landmark = self.landmarks[index]
            return landmark.x, landmark.y

        return None

    def is_hand_detected(self):
        return self.landmarks is not None

    def close(self):
        """Release MediaPipe detector resources."""
        if self.detector is not None:
            self.detector.close()
            self.detector = None
