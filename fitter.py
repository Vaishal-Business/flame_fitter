
import torch
import logging
from config import DEVICE, STAGES
from loss import FittingLoss

class StagedFitter:
    def __init__(self, flame_model, camera, image_data):
        self.flame = flame_model
        self.camera = camera
        self.data = image_data
        self.loss_fn = FittingLoss()
        
        # Move target data to tensors
        self.gt_landmarks = torch.tensor(self.data["landmarks"], dtype=torch.float32, device=DEVICE)
        self.gt_contour = torch.tensor(self.data["contour"], dtype=torch.float32, device=DEVICE)
        self.lm_weights = torch.tensor(self.data["weights"], dtype=torch.float32, device=DEVICE)

    def set_requires_grad(self, stage_name):
        """Freezes and unfreezes parameters based on the current stage."""
        opt_vars = STAGES[stage_name]["optimize"]
        
        # Freeze all first
        for param in self.flame.parameters(): param.requires_grad = False
        for param in self.camera.parameters(): param.requires_grad = False
        
        # Selectively unfreeze
        if "cam" in opt_vars:
            for param in self.camera.parameters(): param.requires_grad = True
        if "pose" in opt_vars:
            self.flame.global_pose.requires_grad = True
            self.flame.jaw_pose.requires_grad = True
        if "shape" in opt_vars:
            self.flame.shape_params.requires_grad = True
        if "exp" in opt_vars:
            self.flame.exp_params.requires_grad = True

    def get_optim_params(self):
        params = []
        for p in self.flame.parameters():
            if p.requires_grad: params.append(p)
        for p in self.camera.parameters():
            if p.requires_grad: params.append(p)
        return params

    def fit(self):
        """Executes the staged optimization schedule strictly on CPU."""
        logging.info("Starting Staged FLAME Optimization...")
        
        for stage_name, stage_cfg in STAGES.items():
            logging.info(f"--- STAGE {stage_name}: Optimizing {stage_cfg['optimize']} ---")
            self.set_requires_grad(stage_name)
            params = self.get_optim_params()
            
            if not params:
                continue

            if stage_cfg["optimizer"] == "Adam":
                optimizer = torch.optim.Adam(params, lr=stage_cfg["lr"])
                self._run_adam(optimizer, stage_cfg["iterations"], stage_name)
            elif stage_cfg["optimizer"] == "LBFGS":
                optimizer = torch.optim.LBFGS(params, lr=stage_cfg["lr"], max_iter=20)
                self._run_lbfgs(optimizer, stage_cfg["iterations"], stage_name)
                
        logging.info("Optimization Complete.")

    def _run_adam(self, optimizer, iterations, stage_name):
        for i in range(iterations):
            optimizer.zero_grad()
            vertices = self.flame()
            proj_verts = self.camera(vertices)
            
            loss, loss_dict = self.loss_fn(proj_verts, self.gt_landmarks, self.gt_contour, self.lm_weights, self.flame)
            
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0 or i == iterations - 1:
                logging.info(f"Stage {stage_name} | Iter {i} | Total Loss: {loss_dict['total']:.2f} | Shape Reg: {loss_dict['shape_reg']:.4f}")

    def _run_lbfgs(self, optimizer, iterations, stage_name):
        for i in range(iterations):
            def closure():
                optimizer.zero_grad()
                vertices = self.flame()
                proj_verts = self.camera(vertices)
                loss, _ = self.loss_fn(proj_verts, self.gt_landmarks, self.gt_contour, self.lm_weights, self.flame)
                loss.backward()
                return loss
                
            optimizer.step(closure)
            
            if i % 10 == 0 or i == iterations - 1:
                with torch.no_grad():
                    vertices = self.flame()
                    proj = self.camera(vertices)
                    loss, l_dict = self.loss_fn(proj, self.gt_landmarks, self.gt_contour, self.lm_weights, self.flame)
                logging.info(f"Stage {stage_name} LBFGS | Iter {i} | Total Loss: {l_dict['total']:.2f}")
