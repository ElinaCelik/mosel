# code from https://github.com/AvivNavon/nash-mtl/tree/main
# and adapted

import torch
import numpy as np
import cvxpy as cp
from utils import num_parameters, model_from_dataset
from ..base import BaseMethod


class NashMTLMethod(BaseMethod):
    def __init__(self, objectives, num_starts, lr, dim, max_norm, update_weights_every, **kwargs):
        self.objectives  = objectives
        self.K           = len(objectives)
        self.num_starts  = num_starts
        self.lr          = lr
        self.max_norm    = max_norm
        self.update_weights_every = update_weights_every

        dim = list(dim)
        self.device = (
            torch.device("cpu") if len(dim) == 1
            else torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.model = model_from_dataset(method='nashmtl', dim=dim, **kwargs).to(self.device)
        self.n_params = num_parameters(self.model)

        self._iter = 0
        self.prvs_alpha = np.ones(self.K, dtype=np.float32)
        self.normalization_factor = np.ones((1,))
        self.init_gtg = np.eye(self.K)
        self.alpha_param = None
        self.prvs_alpha_param = None
        self.G_param = None
        self.normalization_factor_param = None
        self.prob = None

    def model_params(self):
        return self.model.parameters()

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.K,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(shape=(self.K,), value=self.prvs_alpha)
        self.G_param = cp.Parameter(shape=(self.K, self.K), value=self.init_gtg)
        self.normalization_factor_param = cp.Parameter(shape=(1,), value=np.array([1.0]))

        phi_alpha = self._calc_phi_alpha_linearization()
        G_alpha = self.G_param @ self.alpha_param
        constraints = [
            -cp.log(self.alpha_param[i] * self.normalization_factor_param) - cp.log(G_alpha[i]) <= 0
            for i in range(self.K)
        ]
        obj = cp.Minimize(cp.sum(G_alpha) + phi_alpha / self.normalization_factor_param)
        self.prob = cp.Problem(obj, constraints)

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        return prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)

    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value) < 1e-6)
        )

    def solve_optimization(self, gtg: np.ndarray) -> np.ndarray:
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.num_starts):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except Exception:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def step(self, batch):
        self.model.train()

        if self._iter == 0:
            self._init_optim_problem()

        if (self._iter % self.update_weights_every) == 0:
            # compute per-objective gradients wrt all model parameters
            grads = {}
            for i, objective in enumerate(self.objectives):
                self.model.zero_grad()
                logits = self.model(batch)
                batch.update(logits)
                loss = objective(**batch)
                g = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
                g = [gi if gi is not None else torch.zeros_like(p)
                     for gi, p in zip(g, self.model.parameters())]
                grads[i] = torch.cat([torch.flatten(gi) for gi in g])

            G = torch.stack(list(grads.values()))
            GTG = torch.mm(G, G.t())

            self.normalization_factor = torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            self._alpha = torch.from_numpy(alpha.astype(np.float32))

        self._iter += 1

        # weighted forward + backward pass
        self.model.zero_grad()
        logits = self.model(batch)
        batch.update(logits)
        weighted_loss = sum(
            objective(**batch) * self._alpha[i].to(self.device)
            for i, objective in enumerate(self.objectives)
        )
        weighted_loss.backward()

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

        return weighted_loss.item()

    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            return [self.model(batch)]