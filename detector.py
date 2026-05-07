"""
Face Detection module using OpenCV's DNN (Deep Neural Network) module.
Uses a pre-trained Caffe SSD model for robust face detection.
"""

import os
import urllib.request
import cv2
import numpy as np


# Model file URLs (hosted by OpenCV)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

PROTOTXT_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


class FaceDetector:
    """
    Detects faces in frames using OpenCV's DNN module with a Caffe SSD model.
    
    The model is automatically downloaded on first use if not already present.
    """

    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the face detector.

        Args:
            confidence_threshold: Minimum confidence (0-1) to consider a detection valid.
        """
        self.confidence_threshold = confidence_threshold
        self._ensure_model_files()
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        print("[INFO] Face detection DNN model loaded successfully.")

    def _ensure_model_files(self):
        """Download model files if they don't exist."""
        os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.isfile(PROTOTXT_PATH):
            print("[INFO] Downloading deploy.prototxt...")
            urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
            print("[INFO] deploy.prototxt downloaded.")

        if not os.path.isfile(CAFFEMODEL_PATH):
            print("[INFO] Downloading Caffe model (10 MB)... This may take a moment.")
            urllib.request.urlretrieve(CAFFEMODEL_URL, CAFFEMODEL_PATH)
            print("[INFO] Caffe model downloaded.")

    def detect(self, frame):
        """
        Detect faces in a frame.

        Args:
            frame: BGR image (numpy array) from OpenCV.

        Returns:
            List of tuples (x, y, w, h, confidence) for each detected face.
        """
        h, w = frame.shape[:2]

        # Create a blob from the frame for the DNN
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < self.confidence_threshold:
                continue

            # Extract bounding box coordinates (model outputs normalized coords)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face_w = x2 - x1
            face_h = y2 - y1

            if face_w > 0 and face_h > 0:
                faces.append((x1, y1, face_w, face_h, float(confidence)))

        return faces
