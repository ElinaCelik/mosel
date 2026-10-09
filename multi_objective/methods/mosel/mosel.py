import torch
import torch.nn as nn
import numpy as np

from utils import num_parameters, model_from_dataset, circle_points
from ..base import BaseMethod
from models.simple import FullyConnected, MultiLeNet
from models.efficient_net.efficient_net_wrapper import EfficientNetWrapper
from models.resnet.resnet_wrapper import ResNetWrapper
from models.distilbert.distilbert_wrapper import DistilBERTWrapper
from models.distilbert.distilbert_wrapper import TransformerReprNetwork

class Upsampler(nn.Module):

    def __init__(self, K, preference_alignment_network, representation_network,
                 input_dim, dim_z, **kwargs):
        super().__init__()

        self.beta_only = kwargs.pop('beta_only', False)
        self.to_logits_dict = kwargs.pop('to_logits_dict', None)
        self.text = False

        if len(input_dim) == 1:
            self.tabular = True
        elif len(input_dim) == 3:
            self.tabular = False
            self.dim_z = dim_z
            self.film = nn.Linear(K, 2 * dim_z)
        else:
            raise ValueError(
                f"Expected 1 or 3 input dimensions, got {len(input_dim)}"
            )

        if isinstance(representation_network, TransformerReprNetwork):
            self.tabular = False
            self.text = True
            self.dim_z = dim_z
            self.film = nn.Linear(K, 2 * dim_z)


        self.preference_alignment_network = preference_alignment_network
        self.representation_network = representation_network
        self.settings = kwargs
        self.K = K

    def forward(self, batch):
        if self.text:
            return self._forward_text(batch)


        x = batch['data']
        b = x.shape[0]
        a = batch['alpha'].repeat(b, 1)
        z = self.representation_network(x)

        if not self.tabular:
            gamma, beta = self.film(a).chunk(2, dim=1)
            if z.dim() == 4:
                gamma = gamma.view(b, -1, 1, 1)
                beta = beta.view(b, -1, 1, 1)

            
            z = (z + beta) if self.beta_only else (gamma * z + beta)

            logits = self.preference_alignment_network["shared"](z)

            if self.to_logits_dict is not None:
                return self.to_logits_dict(logits)
            else:
                if logits.dim() == 4:
                    logits = logits.flatten(1)
                return dict(
                    logits_l=self.preference_alignment_network["left"](logits),
                    logits_r=self.preference_alignment_network["right"](logits),
                )

        else:
            z = torch.cat([z, a], dim=1)
            return dict(logits=self.preference_alignment_network(z))

    def _forward_text(self, batch):
        input_ids = batch['data']
        attention_mask = batch['attention_mask']
        b = input_ids.shape[0]
        a = batch['alpha'].repeat(b, 1)
 
        # repr network: embeddings + repr transformer blocks -> (B, L, H)
        z = self.representation_network(input_ids, attention_mask)
 
        # FiLM conditioning on the sequence representation
        gamma, beta = self.film(a).chunk(2, dim=1)
        # z is (B, L, H) — broadcast over sequence dim
        gamma = gamma.unsqueeze(1)  # (B, 1, H)
        beta = beta.unsqueeze(1)    # (B, 1, H)
        z = (z + beta) if self.beta_only else (gamma * z + beta)
 
        # cond network: remaining blocks + CLS pool + fc -> (B, num_classes)
        logits = self.preference_alignment_network["shared"](z, attention_mask)
 
        if self.to_logits_dict is not None:
            return self.to_logits_dict(logits)
        else:
            return dict(logits=logits)


    def params_preference_alignment(self):
        params = list(self.preference_alignment_network.parameters())
        if not self.tabular:
            params += list(self.film.parameters())
        return params

    def params_representation(self):
        return list(self.representation_network.parameters())

    def private_params(self):
        params = []
        if hasattr(self.representation_network, 'private_params'):
            params.extend(self.representation_network.private_params())
        if hasattr(self.preference_alignment_network, 'private_params'):
            params.extend(self.preference_alignment_network.private_params())
        return params


def _probe_dim_z(layers, input_shape):
    with torch.no_grad():
        x = torch.zeros(1, *input_shape)
        for layer in layers:
            x = layer(x)
    return x.shape[1]


class MOSELMethod(BaseMethod):

    def __init__(self, objectives, alpha, dim, n_test_rays, lr,
                 scalarization = 'ls', **kwargs):

        self.objectives = objectives
        self.K = len(objectives)
        self.alpha = alpha
        self.n_test_rays = n_test_rays
        self.lr = lr
        self.scalarization = scalarization
        self.u = None

        dim = list(dim)
        if len(dim) == 1 and not kwargs.get('text', False):
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )

        model = model_from_dataset(method='cosmos', dim=dim, **kwargs)

        if isinstance(model, MultiLeNet):
            representation_network = nn.Sequential(
                nn.Conv2d(dim[0], 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten(),
            )

            dim_z = 720

            preference_alignment_shared = nn.Sequential(
                nn.Linear(dim_z, 50),
                nn.ReLU(),
            )

            preference_alignment_network = nn.ModuleDict({
                "shared": preference_alignment_shared,
                "left": nn.Linear(50, 10),
                "right": nn.Linear(50, 10),
            })

        elif isinstance(model, EfficientNetWrapper):
            split = int(kwargs['split_network'])

            efficientnet = nn.Sequential(
                model._conv_stem,
                *model._blocks,
                model._conv_head,
                model._avg_pooling,
                model._dropout,
                nn.Flatten(1),
                model._fc,
            )

            all_children = list(efficientnet.children())
            representation_network = nn.Sequential(*all_children[:split])
            dim_z = _probe_dim_z(all_children[:split], tuple(dim))

            preference_alignment_shared = nn.Sequential(*all_children[split:])
            preference_alignment_network = nn.ModuleDict({"shared": preference_alignment_shared})

        elif isinstance(model, ResNetWrapper):
            split = int(kwargs['split_network'])

            resnet = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                model.avgpool,
                nn.Flatten(1),
                model.fc,
            )

            all_children = list(resnet.children())
            representation_network = nn.Sequential(*all_children[:split])
            dim_z = _probe_dim_z(all_children[:split], tuple(dim))

            preference_alignment_shared = nn.Sequential(*all_children[split:])
            preference_alignment_network = nn.ModuleDict({"shared": preference_alignment_shared})

        elif isinstance(model, DistilBERTWrapper):
            split = int(kwargs['split_network'])
            
            all_layers = list(model.repr_layers)  # save full list first
            
            model.split_after = split
            model.repr_layers = nn.ModuleList(all_layers[:split])
            model.cond_layers = nn.ModuleList(all_layers[split:] if split < 6 else [])
            
            representation_network, preference_alignment_shared, dim_z = model.get_split_modules()
            preference_alignment_network = nn.ModuleDict({"shared": preference_alignment_shared})


        else:
            layers = list(model.f.children())
            split = int(kwargs['split_network'])
            representation_network = nn.Sequential(*layers[:split])
            dim_z = layers[split].in_features

            if len(dim) == 1:
                preference_alignment_network = nn.Sequential(
                    nn.Linear(dim_z + self.K, layers[split].out_features),
                    *layers[(split + 1):]
                )
            else:
                preference_alignment_network = nn.ModuleDict({
                    "shared": nn.Sequential(
                        nn.Linear(dim_z, layers[split].out_features),
                        *layers[(split + 1):]
                    )
                })

        if isinstance(model, DistilBERTWrapper):
            dim = [0]

        if isinstance(model, (ResNetWrapper, EfficientNetWrapper, DistilBERTWrapper)):
            self.model = Upsampler(
                self.K, preference_alignment_network, representation_network,
                dim, dim_z, to_logits_dict=model._to_logits_dict, **kwargs
            ).to(self.device)
        elif len(dim) > 1 and not isinstance(model, MultiLeNet):
            self.model = Upsampler(
                self.K, preference_alignment_network, representation_network,
                dim, dim_z,
                to_logits_dict=lambda logits: dict(logits=logits),
                **kwargs
            ).to(self.device)
        else:
            self.model = Upsampler(
                self.K, preference_alignment_network, representation_network,
                dim, dim_z, **kwargs
            ).to(self.device)


        self.n_params = num_parameters(self.model)
        print("Number of parameters: {}".format(self.n_params))

    def new_epoch(self, e):
        self.current_epoch = e

    def step(self, batch):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
        if isinstance(self.alpha, list):
            batch['alpha'] = torch.from_numpy(
                np.random.dirichlet(self.alpha, 1).astype(np.float32).flatten()
            ).to(self.device)
        elif self.alpha > 0:
            batch['alpha'] = torch.from_numpy(
                np.random.dirichlet([self.alpha for _ in range(self.K)], 1).astype(np.float32).flatten()
            ).to(self.device)
        else:
            raise ValueError(f"Unknown value for alpha: {self.alpha}, expecting list or float > 0.")
        
        self.model.zero_grad()
        logits = self.model(batch)
        batch.update(logits)

        raw_losses = torch.stack([obj(**batch) for obj in self.objectives])

        loss_total = None
        task_losses = []
        for a, task_loss in zip(batch['alpha'], raw_losses):
            if self.scalarization == 'ls':
                loss_total = a * task_loss if loss_total is None else loss_total + a * task_loss
            elif self.scalarization == 'cs':
                task_losses.append(a * task_loss)
        
        if self.scalarization == 'cs':
            mu = 0.01
            weighted_losses = torch.stack(task_losses) 
            if self.u is None:
                self.u = weighted_losses.clone().detach()
            else:
                self.u = torch.min(self.u, weighted_losses.clone().detach()) 
            loss_total = mu * torch.logsumexp((weighted_losses - self.u) / mu, dim=0)

        loss_total.backward()

        return loss_total.item()

    def eval_step(self, batch, test_rays=None):
        self.model.eval()
        logits = []
        with torch.no_grad():
            if test_rays is None:
                test_rays = circle_points(self.n_test_rays, dim=self.K)

            for ray in test_rays:
                ray = torch.from_numpy(ray.astype(np.float32)).to(self.device)
                ray /= ray.sum()
                batch['alpha'] = ray
                logits.append(self.model(batch))
        return logits