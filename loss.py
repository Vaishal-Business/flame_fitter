import torch
import torch.nn as nn
from config import DEVICE, WEIGHTS

class FittingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_loss = nn.MSELoss(reduction='none')

    def forward(self, proj_verts, gt_landmarks, gt_contour, lm_weights, flame_model):
        """
        Computes the staged weighted loss on CPU with numerical stability safeguards.
        """
        # 1. Dense Landmark Loss (ICP-like dynamic association for robustness)
        dist_matrix = torch.cdist(gt_landmarks.unsqueeze(0), proj_verts)
        min_dists, _ = torch.min(dist_matrix, dim=2)
        
        # Apply regional weights and clamp to prevent extreme values
        lm_loss = (min_dists.squeeze() * lm_weights).mean()
        lm_loss = torch.clamp(lm_loss, min=0, max=1e6)

        # 2. Contour / Silhouette Loss
        contour_dist = torch.cdist(gt_contour.unsqueeze(0), proj_verts)
        min_contour_dists, _ = torch.min(contour_dist, dim=2)
        contour_loss = min_contour_dists.mean()
        contour_loss = torch.clamp(contour_loss, min=0, max=1e6)
        
        # 3. Regularizations
        shape_reg = torch.sum(flame_model.shape_params ** 2)
        exp_reg = torch.sum(flame_model.exp_params ** 2)
        pose_reg = torch.sum(flame_model.global_pose ** 2) + torch.sum(flame_model.jaw_pose ** 2)
        
        # Clamp regularization terms
        shape_reg = torch.clamp(shape_reg, min=0, max=1e6)
        exp_reg = torch.clamp(exp_reg, min=0, max=1e6)
        pose_reg = torch.clamp(pose_reg, min=0, max=1e6)
        
        total_loss = (WEIGHTS["landmark"] * lm_loss) + 
                     (WEIGHTS["contour"] * contour_loss) + 
                     (WEIGHTS["shape_reg"] * shape_reg) + 
                     (WEIGHTS["exp_reg"] * exp_reg) + 
                     (WEIGHTS["pose_reg"] * pose_reg)
        
        # Final clamp to prevent infinity
        total_loss = torch.clamp(total_loss, min=0, max=1e6)
                      
        losses_dict = {
            "total": total_loss,
            "landmark": lm_loss.item(),
            "contour": contour_loss.item(),
            "shape_reg": shape_reg.item(),
            "exp_reg": exp_reg.item()
        }
        
        return total_loss, losses_dict