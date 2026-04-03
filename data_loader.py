import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import logging
from config import REGION_WEIGHTS

class DenseLandmarkExtractor:
    def __init__(self, model_path="/content/drive/MyDrive/face_landmarker.task"):
        # Initialize Face Landmarker using the modern Tasks API
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        logging.info("FaceLandmarker task initialized.")

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Proper MediaPipe Image creation for the Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.landmarker.detect(mp_image)

        # Check if any face was detected
        if not result.face_landmarks:
            raise ValueError("No face detected in the image.")

        # PROPER ACCESS: result.face_landmarks is a list of faces
        # Each element in the list is a list of NormalizedLandmark objects
        face_landmarks = result.face_landmarks[0]

        # Convert to pixel coordinates
        landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks], dtype=np.float32)

        # DEFINE OVAL INDICES MANUALLY (Correct way if solutions is missing/outdated)
        # These are the standard 36 indices for the Face Oval in MediaPipe
        oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 
            400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 
            54, 103, 67, 109
        ]
        contour_points = landmarks[oval_indices]

        # Generate weights using safe dictionary access
        weights = np.ones(len(landmarks), dtype=np.float32) * REGION_WEIGHTS.get("default", 1.0)

        # Define indices for specific facial regions (MediaPipe 468/478 standard)
        jaw_indices = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]
        eyes_indices = [33, 133, 362, 263, 468, 473] # Includes Iris if using 478 model
        nose_indices = [1, 2, 98, 327, 168]

        weights[jaw_indices] = REGION_WEIGHTS.get("jaw", 1.0)
        weights[eyes_indices] = REGION_WEIGHTS.get("eyes", 1.0)
        weights[nose_indices] = REGION_WEIGHTS.get("nose", 1.0)

        logging.info(f"Extracted {len(landmarks)} landmarks and {len(contour_points)} contour points.")

        return {
            "image": img,
            "landmarks": landmarks,
            "contour": contour_points,
            "weights": weights,
            "dimensions": (h, w)
        }
