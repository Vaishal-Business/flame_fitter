
import torch

# ==========================================
# STRICTLY CPU ENFORCEMENT
# ==========================================
DEVICE = torch.device("cpu")

# ==========================================
# FLAME MODEL PARAMETERS
# ==========================================
FLAME_MODEL_PATH = "/content/drive/MyDrive/FLAME_202/flame2023_Open.pkl" # Path to downloaded FLAME model
NUM_SHAPE_PARAMS = 100
NUM_EXP_PARAMS = 50

# ==========================================
# OPTIMIZATION STAGES
# ==========================================
# Stage A: Camera & Pose only (Rough alignment)
# Stage B: Shape & Camera (Shape dominant)
# Stage C: Expression & Camera (Tiny expression tweaks)
# Stage D: Final Refinement (Shape, Pose, Camera - Joint LBFGS)

STAGES = {
    "A": {
        "optimize": ["cam", "pose"],
        "iterations": 150,
        "lr": 0.01,
        "optimizer": "Adam"
    },
    "B": {
        "optimize": ["cam", "shape"],
        "iterations": 250,
        "lr": 0.01,
        "optimizer": "Adam"
    },
    "C": {
        "optimize": ["exp"],
        "iterations": 100,
        "lr": 0.005,
        "optimizer": "Adam"
    },
    "D": {
        "optimize": ["cam", "pose", "shape"],
        "iterations": 50,
        "lr": 0.1,
        "optimizer": "LBFGS"
    }
}

# ==========================================
# LOSS WEIGHTS & REGULARIZATION
# ==========================================
WEIGHTS = {
    "landmark": 1.0,
    "contour": 1.5,      # High weight on silhouette for shape accuracy
    "shape_reg": 1e-4,   # Regularize shape heavily to prevent monster faces
    "exp_reg": 1e-3,     # Heavily penalize expression to force shape to do the work
    "pose_reg": 1e-4
}

# Regional Landmark Weights
REGION_WEIGHTS = {
    "jaw": 2.5,          # Vital for shape
    "chin": 2.5,
    "eyes": 1.5,
    "nose": 1.5,
    "mouth": 0.5,        # Low weight, too sensitive to expression
    "default": 1.0
}
