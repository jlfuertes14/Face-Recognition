"""
UI module for drawing overlays, bounding boxes, labels, and HUD elements
on the camera feed using OpenCV drawing primitives.
"""

import cv2
import time


# ── Color Palette (BGR) ─────────────────────────────────────────────
COLOR_GREEN = (0, 220, 80)
COLOR_RED = (0, 0, 230)
COLOR_YELLOW = (0, 220, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_DARK_BG = (30, 30, 30)
COLOR_ACCENT = (255, 160, 0)  # Orange-ish

# ── Fonts ────────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


class UIRenderer:
    """Renders bounding boxes, labels, HUD, and help text on frames."""

    def __init__(self):
        self._fps_start = time.time()
        self._fps_count = 0
        self._fps_display = 0.0

    def draw_face_box(self, frame, x, y, w, h, name="Unknown", confidence=0.0):
        """
        Draw a styled bounding box and label around a detected face.

        Args:
            frame: The image to draw on.
            x, y, w, h: Bounding box coordinates.
            name: Recognized name or "Unknown".
            confidence: Match confidence (0-1).
        """
        is_known = name != "Unknown"
        color = COLOR_GREEN if is_known else COLOR_RED

        # ── Draw corner brackets instead of a full rectangle ──
        corner_len = max(15, min(w, h) // 5)
        thickness = 2

        # Top-left
        cv2.line(frame, (x, y), (x + corner_len, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_len), color, thickness)
        # Top-right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness)
        # Bottom-left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness)
        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

        # ── Label background ──
        if is_known:
            label = f"{name} ({int(confidence * 100)}%)"
        else:
            label = "Unknown"

        font_scale = 0.55
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, font_scale, 1)
        label_y = y - 10 if y - 10 > text_h else y + h + text_h + 10

        # Background pill
        cv2.rectangle(
            frame,
            (x, label_y - text_h - 6),
            (x + text_w + 10, label_y + 4),
            color, -1
        )
        cv2.putText(frame, label, (x + 5, label_y - 2), FONT, font_scale, COLOR_WHITE, 1, cv2.LINE_AA)

    def draw_hud(self, frame, mode, registered_count):
        """
        Draw a heads-up display with FPS, mode, and registered face count.

        Args:
            frame: The image to draw on.
            mode: Current mode string ("DETECTION" or "RECOGNITION").
            registered_count: Number of registered faces in the database.
        """
        h, w = frame.shape[:2]

        # ── Calculate FPS ──
        self._fps_count += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self._fps_display = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_start = time.time()

        # ── Top bar ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "FACE RECOGNITION SYSTEM", (10, 28),
                     FONT_BOLD, 0.6, COLOR_ACCENT, 1, cv2.LINE_AA)

        # FPS
        fps_text = f"FPS: {self._fps_display:.1f}"
        cv2.putText(frame, fps_text, (w - 120, 28),
                     FONT, 0.5, COLOR_YELLOW, 1, cv2.LINE_AA)

        # ── Status bar (bottom) ──
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 36), (w, h), COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

        mode_color = COLOR_GREEN if mode == "RECOGNITION" else COLOR_YELLOW
        cv2.putText(frame, f"Mode: {mode}", (10, h - 12),
                     FONT, 0.5, mode_color, 1, cv2.LINE_AA)

        reg_text = f"Registered: {registered_count}"
        cv2.putText(frame, reg_text, (w - 150, h - 12),
                     FONT, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

    def draw_help(self, frame):
        """Draw keyboard shortcut hints on the right side of the frame."""
        h, w = frame.shape[:2]

        shortcuts = [
            "[R] Register Face",
            "[D] Delete Face",
            "[L] List Faces",
            "[T] Toggle Mode",
            "[Q] Quit",
        ]

        start_y = 70
        for i, text in enumerate(shortcuts):
            y = start_y + i * 22
            cv2.putText(frame, text, (w - 185, y),
                         FONT, 0.4, COLOR_WHITE, 1, cv2.LINE_AA)

    def draw_message(self, frame, message, duration_remaining=0):
        """
        Draw a temporary message banner at the center of the screen.

        Args:
            frame: The image to draw on.
            message: Text to display.
            duration_remaining: Seconds remaining (for fading, optional).
        """
        h, w = frame.shape[:2]

        font_scale = 0.7
        (text_w, text_h), _ = cv2.getTextSize(message, FONT_BOLD, font_scale, 1)

        cx = (w - text_w) // 2
        cy = h // 2

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (cx - 20, cy - text_h - 15), (cx + text_w + 20, cy + 15), COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.putText(frame, message, (cx, cy), FONT_BOLD, font_scale, COLOR_ACCENT, 1, cv2.LINE_AA)

    def draw_registration_guide(self, frame):
        """Draw a guide overlay when in registration mode."""
        h, w = frame.shape[:2]

        # Center circle guide
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 5

        cv2.circle(frame, (center_x, center_y), radius, COLOR_ACCENT, 2)
        cv2.putText(frame, "Position face in circle", (center_x - 110, center_y + radius + 25),
                     FONT, 0.5, COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, "Press ENTER to capture | ESC to cancel",
                     (center_x - 160, center_y + radius + 50),
                     FONT, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
