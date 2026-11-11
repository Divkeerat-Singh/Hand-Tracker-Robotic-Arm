import numpy as np

class GestureInterpreter:
    def __init__(self):
        self.angles = [90.0, 0.0, 0.0]

        self.angle_limits = [
            (0.0, 180.0),
            (-90.0, 90.0),
            (-60.0, 60.0)
        ]

        self.hand_rotation_range = 60.0

        # Current selected joint.
        self.active_joint = None

        # Hand rotation when the current gesture was activated.
        self.neutral_hand_angle = None

        # Joint angle when the current gesture was activated.
        self.neutral_joint_angle = None

    #Geometry
    def get_angle(self, p1, p2):
        """Calculate the 2D angle between two points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        return np.degrees(np.arctan2(-dy, dx))

    def calculate_3d_angle(self, a, b, c):
        """
        Calculate the angle ABC using 3D landmark coordinates.
        This helps identify whether a finger is extended or curled.
        """
        ba = np.array([
            a.x - b.x,
            a.y - b.y,
            a.z - b.z
        ])

        bc = np.array([
            c.x - b.x,
            c.y - b.y,
            c.z - b.z
        ])

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba == 0 or norm_bc == 0:
            return 0.0

        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine = np.clip(cosine, -1.0, 1.0)

        return np.degrees(np.arccos(cosine))

    #Detect Fingers
    def count_fingers(self, landmarks):
        """
        Count extended fingers using the angles at their joints.

        Landmark groups:
        Index:  5, 6, 8
        Middle: 9, 10, 12
        Ring:   13, 14, 16
        Pinky:  17, 18, 20
        """
    
        finger_joints = [
            (5, 6, 8),
            (9, 10, 12),
            (13, 14, 16),
            (17, 18, 20)
        ]

        extended_fingers = 0

        # Detect the four fingers.
        for mcp, pip, tip in finger_joints:
            angle = self.calculate_3d_angle(
                landmarks[mcp],
                landmarks[pip],
                landmarks[tip]
            )

            if angle > 150:
                extended_fingers += 1

        # Detect thumb extension using its joint angle.
        thumb_angle = self.calculate_3d_angle(
            landmarks[1],
            landmarks[2],
            landmarks[4]
        )

        if thumb_angle > 145:
            extended_fingers += 1

        return extended_fingers

    #Hand Rotation
    def get_hand_rotation(self, landmarks, frame_width, frame_height):
        """
        Calculate the palm's orientation in the camera image.

        Uses:
        Landmark 0: Wrist
        Landmark 9: Middle finger MCP

        The angle changes as the hand rotates in the image plane.
        """

        wrist = (
            landmarks[0].x * frame_width,
            landmarks[0].y * frame_height
        )

        middle_mcp = (
            landmarks[9].x * frame_width,
            landmarks[9].y * frame_height
        )

        return self.get_angle(wrist, middle_mcp)

    #Joint Mapping
    def get_active_joint(self, finger_count):
        """
        Map finger gestures to robot joints.

        5 fingers = Shoulder
        2 fingers = Elbow
        1 finger  = Wrist
        """

        if finger_count == 5:
            return 0

        elif finger_count == 2:
            return 1

        elif finger_count == 1:
            return 2

        return None

    #Angle Mapping
    def normalize_angle_difference(self, current, reference):
        """
        Calculate the shortest angular difference.

        This prevents sudden jumps when the angle crosses
        the -180 / +180 degree boundary.
        """
        return (current - reference + 180) % 360 - 180

    def map_hand_rotation(self, rotation_difference, joint_index):
        """
        Convert hand rotation into a target robot joint angle.

        The hand's neutral orientation maps to the joint's
        angle at the moment the gesture was activated.

        Shoulder:
            Absolute angle, 0 to 180 degrees.

        Elbow:
            Relative angle, -90 to +90 degrees.

        Wrist:
            Relative angle, -60 to +60 degrees.
        """

        minimum, maximum = self.angle_limits[joint_index]

        # Calculate how much of the joint's range is available
        # in either direction from its neutral position.
        positive_range = maximum - self.neutral_joint_angle
        negative_range = self.neutral_joint_angle - minimum

        if rotation_difference >= 0:
            movement = (
                rotation_difference / self.hand_rotation_range
            ) * positive_range

        else:
            movement = (
                rotation_difference / self.hand_rotation_range
            ) * negative_range

        target_angle = self.neutral_joint_angle + movement

        return float(np.clip(target_angle, minimum, maximum))

    #Interpretation
    def interpret(self, landmarks, frame_width, frame_height):
        """
        Interpret hand gestures and return target joint angles.

        The selected joint is controlled by hand rotation.
        Other joints retain their previous target angles.
        """

        if landmarks is None:
            return self.angles

        # Identify the gesture.
        finger_count = self.count_fingers(landmarks)

        selected_joint = self.get_active_joint(finger_count)

        # If the gesture is not recognised, retain current angles.
        if selected_joint is None:
            return self.angles

        # Calculate current hand orientation.
        hand_angle = self.get_hand_rotation(
            landmarks,
            frame_width,
            frame_height
        )

        # Detect a change in the selected joint.
        if selected_joint != self.active_joint:

            self.active_joint = selected_joint

            # Establish the current hand orientation as neutral.
            self.neutral_hand_angle = hand_angle

            # Preserve the current target angle of this joint.
            self.neutral_joint_angle = self.angles[selected_joint]

            return self.angles

        # Calculate rotation relative to the neutral orientation.
        rotation_difference = self.normalize_angle_difference(
            hand_angle,
            self.neutral_hand_angle
        )

        # Convert hand rotation into a joint angle.
        target_angle = self.map_hand_rotation(
            rotation_difference,
            selected_joint
        )

        # Update only the selected joint.
        self.angles[selected_joint] = target_angle

        return self.angles

    def get_angles(self):
        return self.angles
