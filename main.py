"""
Face Detection & Recognition System
====================================
Real-time face detection (OpenCV DNN) and recognition (OpenFace DNN embeddings)
using your system camera. No dlib or face_recognition library required.

Controls:
    R     - Register a new face
    D     - Delete a registered face
    L     - List all registered faces
    T     - Toggle between Detection / Recognition mode
    Q/ESC - Quit

Author: Auto-generated
"""

import cv2
import time
import numpy as np

from detector import FaceDetector
from recognizer import FaceRecognizer
from database import FaceDatabase
from ui import UIRenderer


class FaceRecognitionApp:
    """Main application class orchestrating camera, detection, recognition, and UI."""

    def __init__(self):
        print("=" * 55)
        print("   FACE DETECTION & RECOGNITION SYSTEM")
        print("   (Pure OpenCV Deep Learning Pipeline)")
        print("=" * 55)
        print()

        # ── Initialize components ──
        self.detector = FaceDetector(confidence_threshold=0.5)
        self.recognizer = FaceRecognizer(tolerance=0.25)
        self.database = FaceDatabase()
        self.ui = UIRenderer()

        # ── State ──
        self.mode = "RECOGNITION"  # "DETECTION" or "RECOGNITION"
        self.message = ""
        self.message_time = 0
        self.MESSAGE_DURATION = 3.0  # seconds

        # ── Camera ──
        self.cap = None

    def _show_message(self, text):
        """Set a temporary on-screen message."""
        self.message = text
        print(f"[APP] {text}")

    def _start_camera(self):
        """Open the system camera."""
        print("[INFO] Opening camera...")
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("[ERROR] Could not open camera. Please check your camera connection.")
            return False

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("[INFO] Camera opened successfully.")
        return True

    def _process_frame(self, frame):
        """
        Process a single frame: detect faces, optionally recognize them, draw UI.

        Args:
            frame: BGR image from the camera.

        Returns:
            The annotated frame.
        """
        # ── Detect faces ──
        faces = self.detector.detect(frame)

        if self.mode == "RECOGNITION" and faces:
            known_names, known_encodings = self.database.get_all()

            for (x, y, w, h, conf) in faces:
                # Get 128-d embedding from OpenFace model
                encoding = self.recognizer.encode(frame, (x, y, w, h))

                if encoding is not None:
                    name, match_conf = self.recognizer.recognize(
                        encoding, known_encodings, known_names
                    )
                    self.ui.draw_face_box(frame, x, y, w, h, name, match_conf)
                else:
                    self.ui.draw_face_box(frame, x, y, w, h, "Unknown", 0.0)
        else:
            # Detection-only mode: just draw boxes
            for (x, y, w, h, conf) in faces:
                label = f"Face ({int(conf * 100)}%)"
                self.ui.draw_face_box(frame, x, y, w, h, label, conf)

        # ── Draw HUD ──
        self.ui.draw_hud(frame, self.mode, self.database.count())
        self.ui.draw_help(frame)

        # ── Temporary message ──
        if self.message and (time.time() - self.message_time < self.MESSAGE_DURATION):
            self.ui.draw_message(frame, self.message)

        return frame

    def _register_face(self, frame):
        """
        Register a new face from the current frame.
        Captures multiple encodings for better accuracy.

        Args:
            frame: Current camera frame.
        """
        faces = self.detector.detect(frame)

        if not faces:
            self._show_message("No face detected! Please face the camera.")
            return

        if len(faces) > 1:
            self._show_message("Multiple faces detected! Only one face at a time.")
            return

        x, y, w, h, conf = faces[0]

        # Get face encoding
        encoding = self.recognizer.encode(frame, (x, y, w, h))

        if encoding is None:
            self._show_message("Could not encode face. Try again.")
            return

        # Prompt for name in console
        print()
        print("=" * 40)
        name = input("  Enter name for this face: ").strip()
        print("=" * 40)

        if not name:
            self._show_message("Registration cancelled (no name entered).")
            return

        # Register the encoding
        self.database.register(name, encoding)
        self._show_message(f"Registered: {name}")

        # Offer to capture additional samples for better accuracy
        print(f"\n[TIP] For better accuracy, register {name}'s face from different")
        print(f"      angles by pressing 'R' again while facing the camera differently.\n")

    def _delete_face(self):
        """Delete a registered face by name."""
        names = self.database.list_names()

        if not names:
            self._show_message("No faces registered yet.")
            return

        print()
        print("=" * 40)
        print("  Registered faces:")
        for i, name in enumerate(names, 1):
            print(f"    {i}. {name}")
        print()
        name = input("  Enter name to delete (or 'cancel'): ").strip()
        print("=" * 40)

        if not name or name.lower() == "cancel":
            self._show_message("Deletion cancelled.")
            return

        count = self.database.delete(name)
        if count > 0:
            self._show_message(f"Deleted: {name} ({count} encoding(s))")
        else:
            self._show_message(f"Name '{name}' not found.")

    def _list_faces(self):
        """List all registered faces."""
        names = self.database.list_names()

        print()
        print("=" * 40)
        if names:
            print(f"  Registered faces ({len(names)}):")
            for i, name in enumerate(names, 1):
                print(f"    {i}. {name}")
        else:
            print("  No faces registered yet.")
        print("=" * 40)
        print()

        if names:
            self._show_message(f"{len(names)} face(s) registered")
        else:
            self._show_message("No faces registered")

    def run(self):
        """Main application loop."""
        if not self._start_camera():
            return

        print()
        print("─" * 45)
        print("  KEYBOARD CONTROLS:")
        print("  [R] Register face   [D] Delete face")
        print("  [L] List faces      [T] Toggle mode")
        print("  [Q] Quit            [ESC] Quit")
        print("─" * 45)
        print()

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    print("[ERROR] Failed to read from camera.")
                    break

                # Mirror the frame for a natural feel
                frame = cv2.flip(frame, 1)

                # Process and display
                annotated = self._process_frame(frame)
                cv2.imshow("Face Recognition System", annotated)

                # ── Handle keyboard input ──
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:  # Q or ESC
                    print("[INFO] Shutting down...")
                    break

                elif key == ord("r"):
                    self._show_message("Registering... Check console.")
                    self.message_time = time.time()
                    # Re-read a fresh frame right before registering
                    ret2, fresh_frame = self.cap.read()
                    if ret2:
                        fresh_frame = cv2.flip(fresh_frame, 1)
                        self._register_face(fresh_frame)

                elif key == ord("d"):
                    self._delete_face()

                elif key == ord("l"):
                    self._list_faces()

                elif key == ord("t"):
                    if self.mode == "DETECTION":
                        self.mode = "RECOGNITION"
                    else:
                        self.mode = "DETECTION"
                    self._show_message(f"Mode: {self.mode}")
                    self.message_time = time.time()

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")

        finally:
            self._cleanup()

    def _cleanup(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete. Goodbye!")


def main():
    app = FaceRecognitionApp()
    app.run()


if __name__ == "__main__":
    main()
