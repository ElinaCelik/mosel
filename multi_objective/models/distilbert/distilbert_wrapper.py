import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import List, Optional


# Maps dataset name to ordered list of (logit_key, n_classes) per task.
DATASET_TASK_HEADS = {
    'celeba':          [(str(i), 1) for i in range(40)],
    'fairface':        [('race', 7), ('gender', 1), ('age', 9)],
    'civil_comments':  [(str(i), 1) for i in range(7)],
}


"""
Helper modules for Upsampler compatibility 
Transformer layers need attention_mask, but Upsampler calls self.representation_network(x) with a single arg.
These wrappers handle that by accepting either:
  (input_ids, attention_mask)  — when called from the branch setup
  (hidden_states,)             — when attention_mask is stored via set_mask()
"""

class TransformerReprNetwork(nn.Module):
    def __init__(self, embeddings, layers):
        super().__init__()
        self.embeddings = embeddings
        self.layers = layers  
        self._mask = None
        self._mask = None

    def set_mask(self, mask):
        self._mask = mask

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = self._mask
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, attn_mask=attention_mask)[0]
        return x  # (B, L, hidden_dim)


class TransformerCondNetwork(nn.Module):
    def __init__(self, layers, fc):
        super().__init__()
        self.layers = layers  
        self.fc = fc
        self._mask = None

    def set_mask(self, mask):
        self._mask = mask

    def forward(self, x, attention_mask=None):
        if attention_mask is None:
            attention_mask = self._mask
        for layer in self.layers:
            x = layer(x, attn_mask=attention_mask)[0]
        # CLS pooling + linear head
        return self.fc(x[:, 0])


class DistilBERTWrapper(nn.Module):

    def __init__(self, config, backbone, num_classes, split_after=None, **kwargs):
        super().__init__()
        self.config = config
        self.embeddings = backbone.embeddings

        all_layers = list(backbone.transformer.layer)
        n_layers = len(all_layers) 

        if split_after is not None and split_after < n_layers:
            self.repr_layers = nn.ModuleList(all_layers[:split_after])
            self.cond_layers = nn.ModuleList(all_layers[split_after:])
        else:
            self.repr_layers = nn.ModuleList(all_layers)
            self.cond_layers = nn.ModuleList()

        self.split_after = split_after or n_layers
        self.hidden_dim = config.dim  
        self.fc = nn.Linear(config.dim, num_classes)

        # Set by from_name
        self.task_ids = []
        self.task_heads = []


    def get_split_modules(self):
        repr_net = TransformerReprNetwork(self.embeddings, self.repr_layers)
        cond_net = TransformerCondNetwork(self.cond_layers, self.fc)
        return repr_net, cond_net, self.hidden_dim

    def forward_feature_extraction(self, batch):
        x = self.embeddings(batch['input_ids'])
        mask = batch['attention_mask']
        for layer in self.repr_layers:
            x = layer(x, attn_mask=mask)[0]
        x = x[:, 0]  
        return x

    def forward_repr_seq(self, batch):
        x = self.embeddings(batch['input_ids'])
        mask = batch['attention_mask']
        for layer in self.repr_layers:
            x = layer(x, attn_mask=mask)[0]
        return x  

    def forward_cond_seq(self, x, batch):
        mask = batch['attention_mask']
        for layer in self.cond_layers:
            x = layer(x, attn_mask=mask)[0]
        return x[:, 0]

    def forward_linear(self, x):
        x = self.fc(x)
        return self._to_logits_dict(x)

    def forward(self, batch):
        if len(self.cond_layers) > 0:
            x = self.forward_repr_seq(batch)
            x = self.forward_cond_seq(x, batch)
        else:
            x = self.forward_feature_extraction(batch)
        x = self.fc(x)
        return self._to_logits_dict(x)

    def _to_logits_dict(self, x):
        offsets = {}
        offset = 0
        for i, (name, n_classes) in enumerate(self.task_heads):
            offsets[i] = (name, offset, n_classes)
            offset += n_classes

        result = {}
        for task_id in self.task_ids:
            name, start, n_classes = offsets[task_id]
            logits = x[:, start:start + n_classes]
            if n_classes == 1:
                logits = logits[:, 0]
            result['logits_{}'.format(task_id)] = logits
        return result

    @classmethod
    def from_name(cls, model_name, dim, task_ids, dataset='civil_comments',
                  split_after=None, **kwargs):
        if dataset not in DATASET_TASK_HEADS:
            raise ValueError(
                "Unknown dataset '{}'. Register it in DATASET_TASK_HEADS.".format(dataset)
            )

        task_heads = DATASET_TASK_HEADS[dataset]
        num_classes = sum(n for _, n in task_heads)

        config = AutoConfig.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name)

        model = cls(config, backbone, num_classes, split_after=split_after)
        model.task_ids = task_ids
        model.task_heads = task_heads
        return model