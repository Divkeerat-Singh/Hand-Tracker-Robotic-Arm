# renderer.py

import pygame
import cv2
import numpy as np
from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS, TITLE,
    BLACK, CYAN, DARK_CYAN, WHITE, GRAY, GREEN, RED,
    SEGMENT_COLORS, JOINT_RADIUS, SEGMENT_THICKNESS,
    WEBCAM_WIDTH, WEBCAM_HEIGHT
)

class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.large_font = pygame.font.SysFont("consolas", 28, bold=True)

    def draw_background(self):
        self.screen.fill(BLACK)
        # Grid lines for depth
        for x in range(0, WINDOW_WIDTH, 60):
            pygame.draw.line(self.screen, (15, 15, 15), (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, 60):
            pygame.draw.line(self.screen, (15, 15, 15), (0, y), (WINDOW_WIDTH, y))

    def draw_webcam(self, frame):
        if frame is None:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (WEBCAM_WIDTH, WEBCAM_HEIGHT))
        frame_surface = pygame.surfarray.make_surface(
            np.transpose(frame_resized, (1, 0, 2))
        )
        self.screen.blit(frame_surface, (WINDOW_WIDTH - WEBCAM_WIDTH - 20, 20))
        pygame.draw.rect(
            self.screen, CYAN,
            (WINDOW_WIDTH - WEBCAM_WIDTH - 20, 20, WEBCAM_WIDTH, WEBCAM_HEIGHT),
            2
        )

    def draw_arm(self, joints):
        if len(joints) < 2:
            return

        for i in range(len(joints) - 1):
            start = (int(joints[i][0]), int(joints[i][1]))
            end = (int(joints[i+1][0]), int(joints[i+1][1]))

            # Glow effect -- draw thick dim line behind
            pygame.draw.line(self.screen, DARK_CYAN, start, end, SEGMENT_THICKNESS + 6)
            # Main segment
            pygame.draw.line(self.screen, SEGMENT_COLORS[i], start, end, SEGMENT_THICKNESS)

        # Draw joints
        for i, joint in enumerate(joints):
            pos = (int(joint[0]), int(joint[1]))
            pygame.draw.circle(self.screen, DARK_CYAN, pos, JOINT_RADIUS + 4)
            pygame.draw.circle(self.screen, CYAN, pos, JOINT_RADIUS)

        # Draw base
        base = (int(joints[0][0]), int(joints[0][1]))
        pygame.draw.rect(self.screen, CYAN, (base[0] - 20, base[1], 40, 20))

    def draw_base_platform(self, base):
        x, y = int(base[0]), int(base[1])
        pygame.draw.line(self.screen, CYAN, (x - 60, y), (x + 60, y), 3)
        pygame.draw.line(self.screen, GRAY, (x - 80, y + 2), (x + 80, y + 2), 2)

    def draw_hud(self, angles, hand_detected):
        # Title
        title = self.large_font.render("GESTURE ROBOT ARM", True, CYAN)
        self.screen.blit(title, (20, 20))

        # Status
        status_color = GREEN if hand_detected else RED
        status_text = "HAND DETECTED" if hand_detected else "NO HAND"
        status = self.font.render(f"STATUS: {status_text}", True, status_color)
        self.screen.blit(status, (20, 60))

        # Joint angles
        labels = ["SHOULDER", "ELBOW", "WRIST"]
        for i, (label, angle) in enumerate(zip(labels, angles)):
            text = self.font.render(f"{label}: {angle:.1f} deg", True, WHITE)
            self.screen.blit(text, (20, 100 + i * 25))

        # FPS
        fps = self.clock.get_fps()
        fps_text = self.font.render(f"FPS: {fps:.0f}", True, GRAY)
        self.screen.blit(fps_text, (20, WINDOW_HEIGHT - 30))

    def tick(self):
        pygame.display.flip()
        self.clock.tick(FPS)

    def quit(self):
        pygame.quit()