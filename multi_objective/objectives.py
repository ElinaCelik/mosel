import torch


def from_name(names, task_names, settings):
    objectives = {
        'CrossEntropyLoss': CrossEntropyLoss,
        'BinaryCrossEntropyLoss': BinaryCrossEntropyLoss,
        'deo': DEOHyperbolicTangentRelaxation,
    }

    if task_names is not None:
        objective_instances = []
        for i, (n, t) in enumerate(zip(names, task_names)):
            obj = objectives[n](
                label_name=f"labels_{t}",
                logits_name=f"logits_{t}"
            )

            objective_instances.append(obj)
        return objective_instances
    return [objectives[n]() for n in names]


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    
    def __init__(self, label_name='labels', logits_name='logits'):
        super().__init__(reduction='mean')
        self.label_name = label_name
        self.logits_name = logits_name
    

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name].long()
        return super().__call__(logits, labels)


class BinaryCrossEntropyLoss(torch.nn.BCEWithLogitsLoss):
    
    def __init__(self, label_name='labels', logits_name='logits', pos_weight=None):
        if len(label_name) == 1 and label_name[0] == 1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        super().__init__(reduction='mean', pos_weight=torch.Tensor([pos_weight]).to(device) if pos_weight else None)
        self.label_name = label_name
        self.logits_name = logits_name
    

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        if logits.ndim == 2:
            logits = torch.squeeze(logits)
        if labels.dtype != torch.float:
            labels = labels.float()
        return super().__call__(logits, labels)


class DEOHyperbolicTangentRelaxation():

    def __init__(self, label_name='labels', logits_name='logits', s_name='sensible_attribute', c=1):
        self.label_name = label_name
        self.logits_name = logits_name
        self.s_name = s_name
        self.c = c

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        sensible_attribute = kwargs[self.s_name]

        n = logits.shape[0]
        logits = torch.sigmoid(logits)
        s_negative = logits[(sensible_attribute.bool()) & (labels == 1)]
        s_positive = logits[(~sensible_attribute.bool()) & (labels == 1)]

        sum_positive = torch.sum(torch.tanh(self.c * torch.relu(s_positive))) if s_positive.numel() > 0 else s_positive.new_tensor(0.0)
        sum_negative = torch.sum(torch.tanh(self.c * torch.relu(s_negative))) if s_negative.numel() > 0 else s_negative.new_tensor(0.0)

        return 1/n * torch.abs(sum_positive - sum_negative)
