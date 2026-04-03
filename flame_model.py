import torch
import torch.nn as nn
import numpy as np
import pickle
from config import DEVICE, NUM_SHAPE_PARAMS, NUM_EXP_PARAMS

def batch_rodrigues(rot_vecs):
    """CPU-friendly batched rodrigues rotation matrix generation."""
    batch_size = rot_vecs.shape[0]
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    
    rx, ry, rz = rot_dir[:, 0], rot_dir[:, 1], rot_dir[:, 2]
    K = torch.zeros((batch_size, 3, 3), dtype=rot_vecs.dtype, device=DEVICE)
    K[:, 0, 1] = -rz
    K[:, 0, 2] = ry
    K[:, 1, 0] = rz
    K[:, 1, 2] = -rx
    K[:, 2, 0] = -ry
    K[:, 2, 1] = rx
    
    ident = torch.eye(3, dtype=rot_vecs.dtype, device=DEVICE).unsqueeze(0).repeat(batch_size, 1, 1)
    rot_mat = ident + sin.unsqueeze(-1) * K + (1 - cos.unsqueeze(-1)) * torch.bmm(K, K)
    return rot_mat

class FLAME(nn.Module):
    """
    Modular, CPU-only PyTorch implementation of the FLAME 3D head model.
    Compatible with FLAME 2017, 2019, and 2023 versions.
    """
    def __init__(self, model_path):
        super().__init__()
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f, encoding='latin1')
        except Exception as e:
            raise RuntimeError(f"Failed to load FLAME model from {model_path}. Error: {e}")

        # 1. Core templates
        self.register_buffer('v_template', torch.tensor(model_data['v_template'], dtype=torch.float32, device=DEVICE))
        self.register_buffer('faces', torch.tensor(model_data['f'], dtype=torch.long, device=DEVICE))
        
        # 2. Blendshapes (Handling Unified vs Separate Basis)
        all_shapedirs = torch.tensor(model_data['shapedirs'], dtype=torch.float32, device=DEVICE)
        self.register_buffer('shapedirs', all_shapedirs[:, :, :NUM_SHAPE_PARAMS])
        
        if 'exprdirs' in model_data:
            # Older FLAME version
            exprdirs = torch.tensor(model_data['exprdirs'], dtype=torch.float32, device=DEVICE)
        elif 'keyed_exprdirs' in model_data:
            # Standard 2023 version
            exprdirs = torch.tensor(model_data['keyed_exprdirs'], dtype=torch.float32, device=DEVICE)
        else:
            # Unified 2023 version: Expressions start at index 300 of shapedirs
            exprdirs = all_shapedirs[:, :, 300:300 + NUM_EXP_PARAMS]
            
        self.register_buffer('exprdirs', exprdirs[:, :, :NUM_EXP_PARAMS])
        
        # 3. Joint regressor (Handles sparse or dense matrix)
        j_reg = model_data['J_regressor']
        if hasattr(j_reg, 'toarray'):
            j_reg = j_reg.toarray()
        self.register_buffer('J_regressor', torch.tensor(j_reg, dtype=torch.float32, device=DEVICE))
        
        # 4. Skinning weights
        self.register_buffer('weights', torch.tensor(model_data['weights'], dtype=torch.float32, device=DEVICE))
        
        # 5. Trainable Parameters
        self.shape_params = nn.Parameter(torch.zeros(1, NUM_SHAPE_PARAMS, dtype=torch.float32, device=DEVICE))
        self.exp_params = nn.Parameter(torch.zeros(1, NUM_EXP_PARAMS, dtype=torch.float32, device=DEVICE))
        self.global_pose = nn.Parameter(torch.zeros(1, 3, dtype=torch.float32, device=DEVICE))
        self.jaw_pose = nn.Parameter(torch.zeros(1, 3, dtype=torch.float32, device=DEVICE))

    def forward(self):
        """Generates 3D mesh vertices from current parameters."""
        # 1. Add shape and expression blendshapes
        v_shaped = self.v_template.unsqueeze(0) + \
                   torch.einsum('bl,vcl->bvc', self.shape_params, self.shapedirs) + \
                   torch.einsum('bl,vcl->bvc', self.exp_params, self.exprdirs)
                   
        # 2. Regress Joints
        J = torch.einsum('jv,bvc->bjc', self.J_regressor, v_shaped)
        
        # 3. Apply global pose rotation
        # (Using rigid transformation centered at the neck/head joint)
        rot_mats = batch_rodrigues(self.global_pose)
        v_posed = torch.bmm(v_shaped - J[:, 0:1, :], rot_mats.transpose(1, 2)) + J[:, 0:1, :]
        
        return v_posed
