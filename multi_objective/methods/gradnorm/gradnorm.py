import torch
import numpy as np
from utils import num_parameters, model_from_dataset
from ..base import BaseMethod


class GradNormMethod(BaseMethod):
    def __init__(self, objectives, num_starts, lr, dim, asymmetry, **kwargs):
        self.objectives  = objectives
        self.K           = len(objectives)
        self.num_starts  = num_starts
        self.lr          = lr
        self.asymmetry   = asymmetry

        dim = list(dim)
        self.device = (
            torch.device("cpu") if len(dim) == 1
            else torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.model = model_from_dataset(method='gradnorm', dim=dim, **kwargs).to(self.device)
        self.n_params = num_parameters(self.model)

        self.task_weights = torch.nn.Parameter(
            torch.ones(self.K, device=self.device), requires_grad=True
        )
        self.initial_losses = None
        self.weight_optimizer = torch.optim.Adam([self.task_weights], lr=lr)

    def model_params(self):
        return self.model.parameters()

    def step(self, batch):
        self.model.train()

        # Compute unweighted task losses and gradient norms per task
        task_losses_detached = []
        unweighted_norms = []
        for obj in self.objectives:
            self.model.zero_grad()
            logits = self.model(batch)
            loss_i = obj(**{**batch, **logits})
            loss_i.backward()
            g_norm = torch.norm(torch.stack([
                p.grad.norm() for p in self.model.parameters() if p.grad is not None
            ]))
            task_losses_detached.append(loss_i.detach())
            unweighted_norms.append(g_norm.detach())

        task_losses_t = torch.stack(task_losses_detached)
        unweighted_norms = torch.stack(unweighted_norms)

        if self.initial_losses is None:
            self.initial_losses = task_losses_t.clone()

        weighted_norms = self.task_weights * unweighted_norms
        mean_norm = weighted_norms.mean().detach()

        loss_ratio = task_losses_t / (self.initial_losses + 1e-8)
        r_i = loss_ratio / loss_ratio.mean()

        # Target norms are constants; detach to stop gradient flow through them
        target_norms = (mean_norm * r_i ** self.asymmetry).detach()

        # GradNorm loss: update task weights to match gradient norms to targets
        grad_norm_loss = torch.abs(weighted_norms - target_norms).sum()
        self.weight_optimizer.zero_grad()
        grad_norm_loss.backward()
        self.weight_optimizer.step()

        # Renormalize so weights sum to K
        with torch.no_grad():
            self.task_weights.data = (
                self.task_weights.data * self.K / self.task_weights.data.sum()
            ).clamp(min=1e-6)

        # Main backward pass: set model gradients for external optimizer
        self.model.zero_grad()
        logits = self.model(batch)
        task_losses_main = torch.stack([obj(**{**batch, **logits}) for obj in self.objectives])
        total_loss = (self.task_weights.detach() * task_losses_main).sum()
        total_loss.backward()

        return total_loss.item(), 0

    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            return [self.model(batch)]
