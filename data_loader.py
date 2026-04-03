
import cv2
import numpy as np
import mediapipe as mp
import logging
from config import REGION_WEIGHTS

class DenseLandmarkExtractor:
    def __init__(self):
        # Initialize CPU-only MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True, # 478 points including irises
            min_detection_confidence=0.5
        )
        
    def process_image(self, image_path):
        """Loads image, extracts dense landmarks and silhouette contour."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        results = self.face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in the image.")
            
        landmarks_norm = results.multi_face_landmarks[0]
        
        # Convert to pixel coordinates
        landmarks = []
        for lm in landmarks_norm.landmark:
            landmarks.append([lm.x * w, lm.y * h])
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Extract silhouette/contour points using MediaPipe Face Oval topology
        oval_indices = [idx for tup in self.mp_face_mesh.FACEMESH_FACE_OVAL for idx in tup]
        oval_indices = list(set(oval_indices))
        contour_points = landmarks[oval_indices]
        
        # Generate per-landmark weights based on facial regions
        weights = np.ones(len(landmarks), dtype=np.float32) * REGION_WEIGHTS["default"]
        
        # Assign region weights (Using standard MediaPipe index heuristics)
        jaw_indices = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109] 
        eyes_indices = [33, 133, 362, 263] # corners
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
