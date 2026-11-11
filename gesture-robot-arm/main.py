import cv2
import pygame
import sys
from config import WEBCAM_INDEX, WINDOW_WIDTH, WINDOW_HEIGHT
from hand_tracker import HandTracker
from gesture_interpreter import GestureInterpreter
from robot_arm import RobotArm
from renderer import Renderer

def main():
    #Webcam Initialization
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    #modules
    tracker = HandTracker()
    interpreter = GestureInterpreter()
    arm = RobotArm()
    renderer = Renderer()

    print("Running -- show your hand to the camera.")
    print("Press Q to quit.")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                renderer.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    cap.release()
                    renderer.quit()
                    sys.exit()

        # Read webcam frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)

        # Process hand
        frame, landmarks = tracker.process(frame)

        # Interpret gestures
        frame_h, frame_w = frame.shape[:2]
        target_angles = interpreter.interpret(landmarks, frame_w, frame_h)

        # Update arm
        arm.update(target_angles)

        # Render
        renderer.draw_background()
        renderer.draw_webcam(frame)
        renderer.draw_base_platform(arm.base)
        renderer.draw_arm(arm.get_joints())
        renderer.draw_hud(arm.get_angles(), tracker.is_hand_detected())
        renderer.tick()

    cap.release()
    renderer.quit()

if __name__ == "__main__":
    main()