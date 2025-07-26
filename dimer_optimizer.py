# -*- coding: utf-8 -*-
# diffusion_dimer_sophia_v7.py (Sophia clipping for Adam, no clipping for Dimer)

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DimerOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, 
                 dimer_strength=5.0, rotation_interval=10, rotation_lr=0.1,
                 displacement=1e-3, base_optimizer='adam', verbose_logging=False,
                 hessian_beta=0.99, clip_ratio=1.0):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rotation_interval <= 0:
            raise ValueError(f"Invalid rotation interval: {rotation_interval}")
        #if not 1e-6 <= dimer_strength <= 1.0:
        #    raise ValueError(f"Invalid dimer strength: {dimer_strength}")
        if not 1e-6 <= rotation_lr <= 1.0:
            raise ValueError(f"Invalid rotation learning rate: {rotation_lr}")
        if not 0.0 <= hessian_beta < 1.0:
            raise ValueError(f"Invalid hessian beta: {hessian_beta}")
        if clip_ratio <= 0:
            raise ValueError(f"Invalid clip ratio: {clip_ratio}")

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, 
                        hessian_beta=hessian_beta, clip_ratio=clip_ratio)
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
                        'n': torch.randn_like(p.data),
                        'h': torch.zeros_like(p.data)  # Initialize h to zero
                    }
                    if self.base_optimizer == 'adam':
                        state['m'] = torch.zeros_like(p.data)
                        state['v'] = torch.zeros_like(p.data)
                    
                    self.state[p] = state
                    self.state[p]['n'] /= torch.norm(self.state[p]['n'])

        logger.info(f"[DimerOptimizer SOPHIA V7] Initialized with base: {self.base_optimizer.upper()}")

    def _rotate_dimer(self, closure):
        if self.verbose_logging:
            logger.info(f"Step {self.step_count}: Rotating dimer...")

        with torch.no_grad():
            original_params = [p.data.clone() for group in self.param_groups for p in group['params'] if p.requires_grad]
            G1 = [p.grad.data.clone() if p.grad is not None else torch.zeros_like(p.data) for group in self.param_groups for p in group['params'] if p.requires_grad]

            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        p.data.add_(self.state[p]['n'], alpha=self.displacement)
        
        closure()
        
        with torch.no_grad():
            G2 = [p.grad.data.clone() if p.grad is not None else torch.zeros_like(p.data) for group in self.param_groups for p in group['params'] if p.requires_grad]

            for (p, orig_p) in zip(
                [p for group in self.param_groups for p in group['params'] if p.requires_grad],
                original_params
            ):
                p.data.copy_(orig_p)

            for (p, g1, g2) in zip(
                [p for group in self.param_groups for p in group['params'] if p.requires_grad],
                G1, G2
            ):
                n_p = self.state[p]['n']
                grad_diff = g2 - g1
                force_perp = grad_diff - torch.dot(grad_diff.view(-1), n_p.view(-1)) * n_p
                n_new = n_p + self.rotation_lr * force_perp
                self.state[p]['n'] = n_new / (torch.norm(n_new) + 1e-8)

    @torch.no_grad()
    def _orthogonalize_dimer(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                n_p = self.state[p]['n']
                proj_n_on_grad = torch.dot(n_p.view(-1), grad.view(-1)) / (torch.norm(grad.view(-1))**2 + 1e-8) * grad
                self.state[p]['n'] = n_p - proj_n_on_grad
                self.state[p]['n'] /= (torch.norm(self.state[p]['n']) + 1e-8)

    def step(self, closure, recalculate_grad=True):
        if recalculate_grad:
            loss = closure()
        else:
            loss = None

        self.step_count += 1

        if self.step_count % self.rotation_interval == 0:
            self._rotate_dimer(closure)
        
        with torch.no_grad():
            # Disabled orthogonalization to preserve dimer direction for saddle point exploration
            # self._orthogonalize_dimer()

            for group in self.param_groups:
                lr = group['lr']
                beta1, beta2, eps = group['beta1'], group['beta2'], group['eps']
                hessian_beta, clip_ratio = group['hessian_beta'], group['clip_ratio']
                
                for p in group['params']:
                    if p.grad is None:
                        logger.warning(f"Gradient missing for parameter {p}")
                        p.grad = torch.zeros_like(p.data)
                        continue
                    
                    grad = p.grad.data
                    state = self.state[p]
                    n = state['n']
                    grad_proj_n = torch.dot(grad.view(-1), n.view(-1))

                    # --- Translation Step with Sophia-inspired Hessian ---
                    if self.base_optimizer == 'adam':
                        m, v, h = state['m'], state['v'], state['h']
                        # Initialize h with first gradient if zero
                        if self.step_count == 1 and torch.all(h == 0):
                            h.copy_(grad**2)
                        # Update first moment (momentum)
                        m.mul_(beta1).add_(grad, alpha=1 - beta1)
                        # Update second moment (Adam's v, for compatibility)
                        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        # Update Hessian diagonal estimate
                        h.mul_(hessian_beta).add_(grad**2, alpha=1 - hessian_beta)
                        
                        m_hat = m / (1 - beta1 ** self.step_count)
                        h_hat = h / (1 - hessian_beta ** self.step_count)
                        
                        # Sophia-inspired step: use Hessian estimate
                        update = m_hat / (h_hat + eps)
                        # Sophia-style clipping (positive clip_ratio)
                        update = torch.clamp(update, min=-clip_ratio, max=clip_ratio)
                        p.data.add_(update, alpha=-lr)
                    elif self.base_optimizer == 'sgd':
                        h = state['h']
                        if self.step_count == 1 and torch.all(h == 0):
                            h.copy_(grad**2)
                        h.mul_(hessian_beta).add_(grad**2, alpha=1 - hessian_beta)
                        h_hat = h / (1 - hessian_beta ** self.step_count)
                        update = grad / (h_hat + eps)
                        update = torch.clamp(update, min=-clip_ratio, max=clip_ratio)
                        p.data.add_(update, alpha=-lr)
                    else:
                        raise ValueError(f"Unknown base optimizer: {self.base_optimizer}")
                    
                    # --- Dimer Step (unchanged, no clipping) ---
                    dimer_update = -self.dimer_strength * grad_proj_n * n
                    p.data.add_(dimer_update)

        return loss
