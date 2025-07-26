
import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DimerOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, 
                 dimer_strength=0.1, rotation_interval=10, rotation_lr=0.1,
                 displacement=1e-4, base_optimizer='adam', verbose_logging=False):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super(DimerOptimizer, self).__init__(params, defaults)

        self.dimer_strength = dimer_strength
        self.rotation_interval = rotation_interval
        self.rotation_lr = rotation_lr
        self.displacement = displacement
        self.base_optimizer = base_optimizer.lower()
        self.verbose_logging = verbose_logging
        self.step_count = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = {
                        'n': torch.randn_like(p.data)
                    }
                    if self.base_optimizer == 'adam':
                        state['m'] = torch.zeros_like(p.data)
                        state['v'] = torch.zeros_like(p.data)
                    
                    self.state[p] = state
                    self.state[p]['n'] /= torch.norm(self.state[p]['n'])

        logger.info(f"[DimerOptimizer FINAL V4 HY] Initialized with base: {self.base_optimizer.upper()}")

    def _rotate_dimer(self, closure):
        if self.verbose_logging:
            logger.info(f"Step {self.step_count}: Rotating dimer...")

        with torch.no_grad():
            original_params = [p.data.clone() for group in self.param_groups for p in group['params'] if p.requires_grad]
            
            # Recalculate G1
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
                        
                        # Component of grad_diff parallel to n_p
                        force_parallel = torch.dot(grad_diff.view(-1), n_p.view(-1)) * n_p
                        
                        # Component of grad_diff perpendicular to n_p
                        force_perp = grad_diff - force_parallel
                        
                        # Update rule from the paper
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
                    
                    # Project gradient onto the dimer direction
                    grad_proj_n = torch.dot(grad.view(-1), n.view(-1))

                    # --- Dimer Step ---
                    # Move along the dimer direction if it's an ascent direction
                    if grad_proj_n < 0:
                        dimer_update = self.dimer_strength * grad_proj_n * n
                        p.data.add_(dimer_update)

                    # --- Translation Step (Base Optimizer) ---
                    if self.base_optimizer == 'adam':
                        beta1, beta2, eps = group['beta1'], group['beta2'], group['eps']
                        m, v = state['m'], state['v']
                        
                        m.mul_(beta1).add_(grad, alpha=1 - beta1)
                        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        
                        m_hat = m / (1 - beta1 ** self.step_count)
                        v_hat = v / (1 - beta2 ** self.step_count)
                        
                        base_step = m_hat / (torch.sqrt(v_hat) + eps)
                        p.data.add_(base_step, alpha=-lr)
                    else: # Default to SGD
                        p.data.add_(grad, alpha=-lr)
        return
