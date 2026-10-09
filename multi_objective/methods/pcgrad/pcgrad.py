import torch
import numpy as np
from utils import num_parameters, model_from_dataset, circle_points, reset_weights
from ..base import BaseMethod


class PCGradMethod(BaseMethod):
    def __init__(self, objectives, num_starts, lr, dim, **kwargs):
        self.objectives  = objectives
        self.K           = len(objectives)
        self.num_starts  = num_starts
        self.lr          = lr

        dim = list(dim)
        self.device = (
            torch.device("cpu") if len(dim) == 1
            else torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.model = model_from_dataset(method='pcgrad', dim=dim, **kwargs).to(self.device)
        self.n_params = num_parameters(self.model)

        self.ref_vec = torch.Tensor(circle_points(self.num_starts)).to(self.device).float()
        self.pref_idx = -1

    def model_params(self):
        return self.model.parameters()

    def new_epoch(self, e):
        if e == 0:
            self.pref_idx += 1
            reset_weights(self.model)
        self.model.train()

    def log(self):
        return {"train_ray": self.ref_vec[self.pref_idx].cpu().numpy().tolist()}

    def step(self, batch):
        self.model.train()

        weights = self.ref_vec[self.pref_idx]  

        grads = []
        task_losses = []
        for obj in self.objectives:
            self.model.zero_grad()
            out = self.model(batch)
            batch_with_out = {**batch, **out}
            loss = obj(**batch_with_out)
            loss.backward()
            task_losses.append(loss.detach())
            grads.append([
                p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                for p in self.model.parameters()
            ])

        proj_grads = [[g.clone() for g in task_g] for task_g in grads]
        for i in range(self.K):
            for j in range(self.K):
                if i == j:
                    continue
                for k, (gi, gj) in enumerate(zip(proj_grads[i], grads[j])):
                    dot = (gi * gj).sum()
                    if dot < 0:
                        proj_grads[i][k] = gi - dot / (gj.norm() ** 2 + 1e-12) * gj

        self.model.zero_grad()
        for param_idx, p in enumerate(self.model.parameters()):
            p.grad = sum(weights[i].item() * pg[param_idx]
                         for i, pg in enumerate(proj_grads))

        return sum(weights[i].item() * l for i, l in enumerate(task_losses)).item(), task_losses[1].item()

    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            return [self.model(batch)]