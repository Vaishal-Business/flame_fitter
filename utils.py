import os
import cv2
import numpy as np
import torch
import logging

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def save_obj(filename, vertices, faces):
    """Saves 3D mesh to OBJ format."""
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ is 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    logging.info(f"Saved mesh to {filename}")

def draw_wireframe_overlay(image, proj_verts, faces, color=(0, 255, 0), thickness=1):
    """
    CPU-only mesh overlay renderer using OpenCV.
    Draws the projected triangle edges onto the 2D image.
    Handles NaN/Inf values that may result from optimization divergence.
    """
    overlay = image.copy()
    proj_verts_np = proj_verts.detach().cpu().numpy()[0]
    
    # Filter out NaN and Inf values
    valid_mask = np.isfinite(proj_verts_np).all(axis=1)
    if not valid_mask.any():
        logging.warning("No valid projected vertices found (all NaN/Inf). Returning original image.")
        return image
    
    # Clip vertices to valid image bounds
    proj_verts_np = np.clip(proj_verts_np, 0, [image.shape[1]-1, image.shape[0]-1])
    proj_verts = proj_verts_np.astype(np.int32)
    
    faces = faces.detach().cpu().numpy()
    
    # Extract edges
    edges = set()
    for face in faces:
        # Only draw edges where all vertices are valid
        if valid_mask[face[0]] and valid_mask[face[1]] and valid_mask[face[2]]:
            edges.add(tuple(sorted((face[0], face[1]))))
            edges.add(tuple(sorted((face[1], face[2]))))
            edges.add(tuple(sorted((face[2], face[0]))))
        
    for edge in edges:
        pt1 = tuple(proj_verts[edge[0]])
        pt2 = tuple(proj_verts[edge[1]])
        cv2.line(overlay, pt1, pt2, color, thickness)
            
    # Blend with original image
    result = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
    return result

def save_state(flame_model, camera, output_dir):
    """Saves the final optimized parameters."""
    state = {
        "shape_params": flame_model.shape_params.detach().cpu().numpy(),
        "exp_params": flame_model.exp_params.detach().cpu().numpy(),
        "global_pose": flame_model.global_pose.detach().cpu().numpy(),
        "camera_scale": camera.scale.detach().cpu().numpy(),
        "camera_trans": camera.trans.detach().cpu().numpy()
    }
    path = os.path.join(output_dir, "fitted_params.npy")
    np.save(path, state)
    logging.info(f"Saved optimized parameters to {path}")
