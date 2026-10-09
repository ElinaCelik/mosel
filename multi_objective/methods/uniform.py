import torch

from utils import model_from_dataset
from .base import BaseMethod


class UniformScalingMethod(BaseMethod):

    def __init__(self, objectives, dim, **kwargs):
        self.objectives = objectives
        self.J = len(objectives)
        if len(dim) == 1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model_from_dataset(method='uniform_scaling', dim=dim, **kwargs).to(self.device)


    def model_params(self):
        return list(self.model.parameters())

    
    def new_epoch(self, e):
        self.model.train()


    def step(self, batch):
        batch.update(self.model(batch))
        loss = sum([1/self.J * o(**batch) for o in self.objectives])
        loss.backward()
        return loss.item()


    def eval_step(self, batch, test_rays=None):
        self.model.eval()
        with torch.no_grad():
            return[self.model(batch)]