
import os
import argparse
import torch
import cv2

from config import FLAME_MODEL_PATH
from data_loader import DenseLandmarkExtractor
from flame_model import FLAME
from camera import WeakPerspectiveCamera
from fitter import StagedFitter
from utils import setup_logger, save_obj, draw_wireframe_overlay, save_state
import logging

def main(image_path, output_dir):
    setup_logger()
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Enforce CPU
    torch.manual_seed(42)
    logging.info("Forcing execution on CPU. Ignoring CUDA entirely.")

    # 2. Extract Data
    logging.info(f"Loading image: {image_path}")
    extractor = DenseLandmarkExtractor()
    image_data = extractor.process_image(image_path)
    
    h, w = image_data["dimensions"]
    
    # 3. Initialize Models
    if not os.path.exists(FLAME_MODEL_PATH):
        logging.error(f"FLAME model not found at {FLAME_MODEL_PATH}. Please download generic_model.pkl.")
        return
        
    flame = FLAME(FLAME_MODEL_PATH)
    
    # Initialize camera. Scale estimation heuristic based on face size vs image size
    initial_scale = w / 250.0 
    initial_trans = [w / 2.0, h / 2.0]
    camera = WeakPerspectiveCamera(scale_init=initial_scale, trans_init=initial_trans)

    # 4. Fit Model
    fitter = StagedFitter(flame, camera, image_data)
    fitter.fit()

    # 5. Export and Visualize
    logging.info("Exporting results...")
    
    with torch.no_grad():
        final_vertices = flame()
        proj_verts = camera(final_vertices)
        
    # Save OBJ
    obj_path = os.path.join(output_dir, "fitted_mesh.obj")
    save_obj(obj_path, final_vertices[0].numpy(), flame.faces)
    
    # Save Overlay Image (CPU Rasterization substitute)
    overlay_img = draw_wireframe_overlay(image_data["image"], proj_verts, flame.faces)
    overlay_path = os.path.join(output_dir, "overlay.jpg")
    cv2.imwrite(overlay_path, overlay_img)
    logging.info(f"Saved visualization to {overlay_path}")
    
    # Save numerical parameters
    save_state(flame, camera, output_dir)
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU-only FLAME Shape Fitting Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input 2D front-facing image")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    args = parser.parse_args()
    
    main(args.image, args.output)
