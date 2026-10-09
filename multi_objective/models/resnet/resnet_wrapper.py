import torch
from .resnet import ResNet, BasicBlock, Bottleneck, model_urls
from .utils import load_state_dict_from_url
from typing import Type, Any, Union, List


# Maps dataset name to ordered list of (logit_key, n_classes) per task.
# CelebA: 40 binary attributes
# FairFace: 3 multi-class tasks
DATASET_TASK_HEADS = {
    'celeba':   [(str(i), 1) for i in range(40)],
    'fairface': [('race', 7), ('gender', 1), ('age', 9)],
}


class ResNetWrapper(ResNet):

    # this is required for approximate mgda
    def forward_feature_extraction(self, batch):
        x = batch['data']
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_linear(self, x):
        x = self.fc(x)
        return self._to_logits_dict(x)

    def forward(self, batch):
        x = self.forward_feature_extraction(batch)
        x = self.fc(x)
        return self._to_logits_dict(x)

    def _to_logits_dict(self, x):
        """
        Split fc output into per-task logits using instance-level task_heads/task_ids.
        """
        # Pre-compute slice boundaries across all heads
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
                logits = logits[:, 0]       # (B,) for binary tasks
            result['logits_{}'.format(task_id)] = logits
        return result

    @classmethod
    def from_name(cls, model_name, dim, task_ids, dataset='celeba', **override_params):
        if dataset not in DATASET_TASK_HEADS:
            raise ValueError(
                "Unknown dataset '{}'. Register it in DATASET_TASK_HEADS.".format(dataset)
            )

        task_heads = DATASET_TASK_HEADS[dataset]

        # fc output covers ALL heads so offsets are always consistent
        num_classes = sum(n for _, n in task_heads)

        model = resnet34(
            pretrained=True,
            progress=False,
            in_channels=dim[0],
            num_classes=num_classes,
        )

        # Set as instance attributes — avoids class-level mutation bugs
        model.task_ids   = task_ids
        model.task_heads = task_heads

        return model


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNetWrapper:
    kwargs.pop('norm_layer', None)
    model = ResNetWrapper(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        in_channels = kwargs.get('in_channels', 3)
        if in_channels != 3:
            state_dict['conv1.weight'] = torch.nn.parameter.Parameter(
                torch.ones((64, in_channels, 7, 7))
            )
        state_dict = {
            k: v for k, v in state_dict.items()
            if 'fc' not in k
        }
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)