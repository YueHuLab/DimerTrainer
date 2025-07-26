
import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DimerSgdOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, 
                 dimer_strength=0.1, rotation_interval=10, rotation_lr=0.1,
                 displacement=1e-4, verbose_logging=False):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr)
        super(DimerSgdOptimizer, self).__init__(params, defaults)

        self.dimer_strength = dimer_strength
        self.rotation_interval = rotation_interval
        self.rotation_lr = rotation_lr
        self.displacement = displacement
        self.verbose_logging = verbose_logging
        self.step_count = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p] = {'n': torch.randn_like(p.data)}
                    self.state[p]['n'] /= torch.norm(self.state[p]['n'])

        logger.info(f"[DimerSgdOptimizer] Initialized with base: SGD")

    def _rotate_dimer(self, closure):
        if self.verbose_logging:
            logger.info(f"Step {self.step_count}: Rotating dimer...")

        with torch.no_grad():
            original_params = [p.data.clone() for group in self.param_groups for p in group['params'] if p.requires_grad]
            closure()
            G1 = [p.grad.data.clone() if p.grad is not None else torch.zeros_like(p.data) for group in self.param_groups for p in group['params'] if p.requires_grad]

            param_idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        p.data.add_(self.state[p]['n'], alpha=self.displacement)
                        param_idx += 1
        
        closure()
        
        with torch.no_grad():
            G2 = [p.grad.data.clone() if p.grad is not None else torch.zeros_like(p.data) for group in self.param_groups for p in group['params'] if p.requires_grad]

            param_idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        p.data.copy_(original_params[param_idx])
                        param_idx += 1

            param_idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        n_p = self.state[p]['n']
                        grad_diff = (G2[param_idx] - G1[param_idx]) / self.displacement
                        force_parallel = torch.dot(grad_diff.view(-1), n_p.view(-1)) * n_p
                        force_perp = grad_diff - force_parallel
                        n_new = n_p - 2 * force_parallel / torch.norm(force_parallel)
                        n_new = n_new + self.rotation_lr * force_perp
                        self.state[p]['n'] = n_new / (torch.norm(n_new) + 1e-8)
                        param_idx += 1

    @torch.no_grad()
    def step(self, closure):
        self.step_count += 1

        if self.step_count % self.rotation_interval == 0:
            self._rotate_dimer(closure)
        else:
            closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                for p in group['params']:
                    if p.grad is None: continue
                    
                    grad = p.grad.data
                    state = self.state[p]
                    n = state['n']
                    
                    grad_proj_n = torch.dot(grad.view(-1), n.view(-1))

                    if grad_proj_n < 0:
                        dimer_update = self.dimer_strength * grad_proj_n * n
                        p.data.add_(dimer_update)

                    # --- Translation Step (SGD) ---
                    p.data.add_(grad, alpha=-lr)
        return
