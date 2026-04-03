# CPU-Only FLAME Shape Fitter

A highly robust, production-quality pipeline for fitting the 3D FLAME head model to a single 2D front-facing image. 

This project strictly enforces **CPU-only execution** and prioritizes **identity/shape geometry** over texture or realistic appearance. It relies on dense landmarks and facial silhouettes (via MediaPipe) mapped against the 3D topology using a dynamic matching algorithm to avoid fragile vertex-mapping files.

## Features
- **Zero GPU Dependency**: No CUDA, no PyTorch3D, runs purely on standard CPU wheels.
- **Staged OptimizationSchedule**: Rough Cam $\to$ Shape-Dominant $\to$ Expression Fixes $\to$ LBFGS Joint Refinement.
- **Dense Landmark & Silhouette Fitting**: Utilizes 468 landmarks + outer face oval contour.
- **Region-weighted losses**: Emphasizes structural anchors like the jaw, chin, and eye sockets.
- **CPU Wireframe Renderer**: Employs an OpenCV-based projection overlay for diagnostics.

## Setup Instructions

1. **Install Dependencies**
Ensure you are using a standard Python environment (Python 3.9+ recommended).
```bash
pip install -r requirements.txt
