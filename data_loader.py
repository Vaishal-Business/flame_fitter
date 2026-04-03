import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks import BaseOptions
import logging
from config import REGION_WEIGHTS

class DenseLandmarkExtractorTask:
    def __init__(self, model_path="/content/drive/MyDrive/face_landmarker.task"):
        # Initialize MediaPipe Face Landmarker Task
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        logging.info("FaceLandmarker task initialized.")

    def process_image(self, image_path):
        """Loads image, extracts dense landmarks and silhouette contour."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Convert to MediaPipe Image
        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=img_rgb)
        
        # Run face landmark detection
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            raise ValueError("No face detected in the image.")

        landmarks_norm = result.face_landmarks[0].landmark

        # Convert normalized landmarks to pixel coordinates
        landmarks = np.array([[lm.x * w, lm.y * h] for lm in landmarks_norm], dtype=np.float32)

        # Extract silhouette/contour points using Face Oval indices
        oval_indices = [idx for tup in vision.FACEMESH_FACE_OVAL for idx in tup]
        oval_indices = list(set(oval_indices))
        contour_points = landmarks[oval_indices]

        # Generate per-landmark weights based on facial regions
        weights = np.ones(len(landmarks), dtype=np.float32) * REGION_WEIGHTS["default"]

        # Assign region weights
        jaw_indices = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        eyes_indices = [33, 133, 362, 263]  # corners
        nose_indices = [1, 2, 98, 327]

        weights[jaw_indices] = REGION_WEIGHTS["jaw"]
        weights[eyes_indices] = REGION_WEIGHTS["eyes"]
        weights[nose_indices] = REGION_WEIGHTS["nose"]

        logging.info(f"Extracted {len(landmarks)} dense landmarks and {len(contour_points)} contour points.")

        return {
            "image": img,
            "image_rgb": img_rgb,
            "landmarks": landmarks,
            "contour": contour_points,
            "weights": weights,
            "dimensions": (h, w)
        }
