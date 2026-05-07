"""
Face Recognition module using OpenCV's DNN module with the OpenFace model.
Generates 128-dimensional face embeddings and matches them via cosine similarity.

This is a pure OpenCV approach — no dlib or face_recognition library needed.
Uses the OpenFace nn4.small2.v1 model (a Torch-based deep neural network).
"""

import os
import urllib.request
import cv2
import numpy as np


# ── Model paths ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OPENFACE_MODEL_URL = "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7"
OPENFACE_MODEL_PATH = os.path.join(MODEL_DIR, "openface_nn4.small2.v1.t7")


class FaceRecognizer:
    """
    Generates 128-d face embeddings using the OpenFace DNN model and
    matches faces via cosine distance.
    """

    def __init__(self, tolerance=0.25):
        """
        Initialize the recognizer.

        Args:
            tolerance: Maximum cosine distance to consider a match.
                       Lower = stricter. Range: 0.2 (strict) to 0.4 (lenient).
        """
        self.tolerance = tolerance
        self._ensure_model()
        self.net = cv2.dnn.readNetFromTorch(OPENFACE_MODEL_PATH)
        print("[INFO] OpenFace embedding model loaded successfully.")

    def _ensure_model(self):
        """Download the OpenFace model if not present."""
        os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.isfile(OPENFACE_MODEL_PATH):
            print("[INFO] Downloading OpenFace model (~31 MB)... This may take a moment.")
            urllib.request.urlretrieve(OPENFACE_MODEL_URL, OPENFACE_MODEL_PATH)
            print("[INFO] OpenFace model downloaded.")

    def encode(self, frame, face_bbox):
        """
        Generate a 128-d embedding for a face region.

        Args:
            frame: BGR image (numpy array).
            face_bbox: Tuple (x, y, w, h) bounding box of the face.

        Returns:
            128-d numpy array (L2 normalized), or None if encoding fails.
        """
        x, y, w, h = face_bbox

        # Extract and prepare the face ROI
        face_roi = frame[y:y + h, x:x + w]

        if face_roi.size == 0:
            return None

        # The OpenFace model expects a 96x96 input with mean subtraction
        blob = cv2.dnn.blobFromImage(
            face_roi,
            scalefactor=1.0 / 255.0,
            size=(96, 96),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        embedding = self.net.forward()

        # Flatten and L2-normalize
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def recognize(self, encoding, known_encodings, known_names):
        """
        Match a face embedding against the database of known faces
        using cosine distance with Euclidean distance verification.

        Args:
            encoding: 128-d numpy array for the face to identify.
            known_encodings: List of 128-d numpy arrays of registered faces.
            known_names: List of names corresponding to known_encodings.

        Returns:
            Tuple of (name, confidence) where confidence is 1 - distance.
            Returns ("Unknown", 0.0) if no match is found.
        """
        if not known_encodings or encoding is None:
            return ("Unknown", 0.0)

        # Compute cosine distances AND Euclidean distances
        cos_distances = []
        euc_distances = []
        for known_enc in known_encodings:
            cos_sim = np.dot(encoding, known_enc)
            cos_dist = 1.0 - cos_sim
            euc_dist = np.linalg.norm(encoding - known_enc)
            cos_distances.append(cos_dist)
            euc_distances.append(euc_dist)

        cos_distances = np.array(cos_distances)
        euc_distances = np.array(euc_distances)
        best_idx = np.argmin(cos_distances)
        best_cos_dist = cos_distances[best_idx]
        best_euc_dist = euc_distances[best_idx]

        # BOTH checks must pass to prevent false positives:
        # 1. Cosine distance must be below tolerance (default 0.25)
        # 2. Euclidean distance must be below 0.85
        if best_cos_dist <= self.tolerance and best_euc_dist <= 0.85:
            confidence = round(1.0 - best_cos_dist, 2)
            return (known_names[best_idx], confidence)

        return ("Unknown", 0.0)
