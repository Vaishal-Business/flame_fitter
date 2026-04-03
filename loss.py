
import torch
import torch.nn as nn
from config import DEVICE, WEIGHTS

class FittingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_loss = nn.MSELoss(reduction='none')

    def forward(self, proj_verts, gt_landmarks, gt_contour, lm_weights, flame_model):
        """
        Computes the staged weighted loss on CPU.
        """
        # 1. Dense Landmark Loss (ICP-like dynamic association for robustness)
        # Since we lack a perfect 468-vertex map, we find the closest projected vertices.
        # This is a safe CPU approach for contour and dense points.
        
        dist_matrix = torch.cdist(gt_landmarks.unsqueeze(0), proj_verts) # (1, N_lmdks, V)
        min_dists, _ = torch.min(dist_matrix, dim=2) # (1, N_lmdks)
        
        # Apply regional weights
        lm_loss = (min_dists.squeeze() * lm_weights).mean()

        # 2. Contour / Silhouette Loss
        # Force outer mesh boundary to match the face oval extracted from MediaPipe
        contour_dist = torch.cdist(gt_contour.unsqueeze(0), proj_verts)
        min_contour_dists, _ = torch.min(contour_dist, dim=2)
        contour_loss = min_contour_dists.mean()
        
        # 3. Regularizations
        shape_reg = torch.sum(flame_model.shape_params ** 2)
        exp_reg = torch.sum(flame_model.exp_params ** 2)
        pose_reg = torch.sum(flame_model.global_pose ** 2) + torch.sum(flame_model.jaw_pose ** 2)
        
        total_loss = (WEIGHTS["landmark"] * lm_loss) + \
                     (WEIGHTS["contour"] * contour_loss) + \
                     (WEIGHTS["shape_reg"] * shape_reg) + \
                     (WEIGHTS["exp_reg"] * exp_reg) + \
                     (WEIGHTS["pose_reg"] * pose_reg)
                     
        losses_dict = {
            "total": total_loss,
            "landmark": lm_loss.item(),
            "contour": contour_loss.item(),
            "shape_reg": shape_reg.item(),
            "exp_reg": exp_reg.item()
        }
        
        return total_loss, losses_dict
