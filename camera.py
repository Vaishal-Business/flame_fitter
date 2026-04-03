
import torch
import torch.nn as nn
from config import DEVICE

class WeakPerspectiveCamera(nn.Module):
    """
    CPU-friendly weak perspective camera.
    Proj = s * R * X + t
    """
    def __init__(self, scale_init=1.0, trans_init=(0.0, 0.0)):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_init], dtype=torch.float32, device=DEVICE))
        self.trans = nn.Parameter(torch.tensor([trans_init], dtype=torch.float32, device=DEVICE))

    def forward(self, vertices):
        """
        vertices: (B, N, 3) 3D points
        returns: (B, N, 2) 2D projected points
        """
        # Weak perspective assumes orthographic projection scaled and translated
        # Drop Z coordinate for projection
        x_xy = vertices[:, :, :2]
        
        # Apply scale and translation
        projected = self.scale * x_xy + self.trans.unsqueeze(1)
        return projected
