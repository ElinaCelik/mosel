import torch

from abc import abstractmethod

import objectives as obj


def from_objectives(objectives, settings):
    scores = {
        obj.CrossEntropyLoss: CrossEntropy,
        obj.BinaryCrossEntropyLoss: BinaryCrossEntropy,
        obj.DEOHyperbolicTangentRelaxation: DEO,
    }

    score_instances = []

    for o in objectives:
        score_class = scores[o.__class__]
        score = score_class(o.label_name, o.logits_name)

        score_instances.append(score)

    return score_instances

class BaseScore():

    def __init__(self, label_name='labels', logits_name='logits', s_name = 'sensible_attribute'):
        super().__init__()
        self.label_name = label_name
        self.logits_name = logits_name
        self.s_name = s_name


    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError()


class CrossEntropy(BaseScore):

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        with torch.no_grad():
            mcr = 1.0 - (torch.argmax(logits, dim=1) == labels).float().mean().item()
            return torch.nn.functional.cross_entropy(logits, labels.long(), reduction='mean').item(), mcr


class BinaryCrossEntropy(BaseScore):
    
    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]

        if len(logits.shape) > 1 and logits.shape[1] == 1:
            logits = torch.squeeze(logits)

        with torch.no_grad():
            preds = (logits > 0).long()
            mcr = 1.0 - (preds == labels).float().mean().item()
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction='mean').item(), mcr


class DEO(BaseScore):
    """Difference in Equality of Opportunity"""

    def __call__(self, **kwargs):
        logits = kwargs[self.logits_name]
        labels = kwargs[self.label_name]
        sensible_attribute = kwargs['sensible_attribute']

        with torch.no_grad():
            n = logits.shape[0]
            logits_s_negative = logits[(sensible_attribute.bool()) & (labels == 1)]
            logits_s_positive = logits[(~sensible_attribute.bool()) & (labels == 1)]

            return (1/n * torch.abs(torch.sum(logits_s_negative > 0) - torch.sum(logits_s_positive > 0))).cpu().item()
