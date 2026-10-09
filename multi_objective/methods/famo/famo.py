import torch
import torch.nn.functional as F
from utils import num_parameters, model_from_dataset
from ..base import BaseMethod


class FAMOMethod(BaseMethod):
    def __init__(self, objectives, num_starts, lr, dim, w_lr, w_weight_decay, **kwargs):
        self.objectives = objectives
        self.K          = len(objectives)
        self.num_starts = num_starts
        self.lr         = lr

        dim = list(dim)
        self.device = (
            torch.device("cpu") if len(dim) == 1
            else torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.model = model_from_dataset(method='famo', dim=dim, **kwargs).to(self.device)
        self.n_params = num_parameters(self.model)

        self.min_losses = torch.zeros(self.K, device=self.device)
        self.w = torch.zeros(self.K, device=self.device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=w_weight_decay)
        self.prev_loss = None

    def model_params(self):
        return self.model.parameters()

    def step(self, batch):
        self.model.train()

        # compute all per-task losses
        logits = self.model(batch)
        task_losses = torch.stack([obj(**{**batch, **logits}) for obj in self.objectives])
        curr_losses = task_losses.detach()

        # Update running per-task minimums
        with torch.no_grad():
            self.min_losses = torch.min(self.min_losses, curr_losses)

        # Update weight logits w using log-space delta between previous and current losses
        if self.prev_loss is not None:
            delta = ((self.prev_loss - self.min_losses + 1e-8).log() - (curr_losses    - self.min_losses + 1e-8).log())
            with torch.enable_grad():
                d = torch.autograd.grad(F.softmax(self.w, dim=-1), self.w, grad_outputs=delta.detach())[0]
            self.w_opt.zero_grad()
            self.w.grad = d
            self.w_opt.step()

        self.prev_loss = curr_losses

        # FAMO weighted loss: L = sum_j z_j * log(D_j) / c
        z = F.softmax(self.w.detach(), dim=-1)
        D = task_losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        loss.backward()

        return loss.item(), 0

    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            return [self.model(batch)]
