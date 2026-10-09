import torch
from models.efficient_net.model import EfficientNet


DATASET_TASK_HEADS = {
    'celeba':   [(str(i), 1) for i in range(40)],
    'fairface': [('race', 7), ('gender', 2), ('age', 9)],
}


class EfficientNetWrapper(EfficientNet):

    # this is required for approximate mgda
    def forward_feature_extraction(self, batch):
        x = batch['data']
        x = self.extract_features(x)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward_linear(self, x):
        x = self._dropout(x)
        x = self._fc(x)
        return self._to_logits_dict(x)

    def forward(self, batch):
        x = batch['data']
        x = super().forward(x)
        return self._to_logits_dict(x)

    def _to_logits_dict(self, x):
        """
        Split fc output into per-task logits using instance-level task_heads/task_ids.
        Offsets are computed across ALL heads so they are stable regardless of
        which task_ids are active.
        """
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

    def private_params(self):
        return [n for n, p in self.named_parameters() if 'task_layers' not in n]

    @classmethod
    def from_pretrained(cls, dim, task_ids, model_name, dataset='celeba', **override_params):
        if dataset not in DATASET_TASK_HEADS:
            raise ValueError(
                "Unknown dataset '{}'. Register it in DATASET_TASK_HEADS.".format(dataset)
            )

        task_heads = DATASET_TASK_HEADS[dataset]
        num_classes = sum(n for _, n in task_heads)

        cls.my_in_channels = dim[0]

        model = super().from_pretrained(
            model_name,
            in_channels=dim[0],
            num_classes=num_classes,
        )

        # Set as instance attributes to avoid class-level mutation bugs
        model.task_ids   = task_ids
        model.task_heads = task_heads

        return model