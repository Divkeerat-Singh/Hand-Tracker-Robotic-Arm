import numpy as np

from config import (
    SEGMENT_LENGTHS,
    SMOOTHING_FACTOR,
    ARM_BASE_X,
    ARM_BASE_Y
)


class RobotArm:
    def __init__(self):
        self.base = (ARM_BASE_X, ARM_BASE_Y)
        self.segment_lengths = SEGMENT_LENGTHS

        self.angles = [90.0, 0.0, 0.0]

        self.angle_limits = [
            (0.0, 180.0),    # Shoulder
            (-90.0, 90.0),   # Elbow
            (-60.0, 60.0)    # Wrist
        ]

        self.joints = []
        self.compute_joints()

    def clamp_angles(self, target_angles):
        """Keep target angles within the permitted joint limits."""
        return [
            float(np.clip(angle, minimum, maximum))
            for angle, (minimum, maximum)
            in zip(target_angles, self.angle_limits)
        ]

    def smooth(self, target_angles):
        """Gradually move current angles towards target angles."""
        target_angles = self.clamp_angles(target_angles)

        for i in range(len(self.angles)):
            self.angles[i] += (
                SMOOTHING_FACTOR *
                (target_angles[i] - self.angles[i])
            )

        self.angles = self.clamp_angles(self.angles)

    def update(self, target_angles):
        """Update the robot arm using new target joint angles."""
        self.smooth(target_angles)
        self.compute_joints()

    def compute_joints(self):
        """
        Calculate joint positions using cumulative angles.

        Segment 1 orientation = shoulder
        Segment 2 orientation = shoulder + elbow
        Segment 3 orientation = shoulder + elbow + wrist
        """
        self.joints = [self.base]

        cumulative_angle = 0.0

        for i, length in enumerate(self.segment_lengths):
            cumulative_angle += self.angles[i]

            rad = np.radians(cumulative_angle)

            previous_joint = self.joints[-1]

            next_x = (
                previous_joint[0] +
                length * np.cos(rad)
            )

            next_y = (
                previous_joint[1] -
                length * np.sin(rad)
            )

            self.joints.append((next_x, next_y))

    def get_joints(self):
        return self.joints

    def get_angles(self):
        return self.angles
