import os
import torch
import numpy as np
import pandas as pd

from torch.utils import data

# Follows the same interface as the CelebA loader from
# https://github.com/intel-isl/MultiObjectiveOptimization


IDENTITY_COLS = [
    'male', 'female', 'transgender', 'other_gender',
    'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
    'other_sexual_orientation', 'christian', 'jewish', 'muslim',
    'hindu', 'buddhist', 'atheist', 'other_religion',
    'black', 'white', 'asian', 'latino', 'other_race_or_ethnicity',
    'physical_disability', 'intellectual_or_learning_disability',
    'psychiatric_or_mental_illness', 'other_disability',
]


class CivilComments(data.Dataset):

    _df_cache = None  # read CSV once across splits

    def __init__(self, split, task_ids=[], root='data/civil_comments',
                 model_name='distilbert-base-uncased', max_length=128,
                 threshold=0.5, max_samples=None, augmentations=None, **kwargs):
        """__init__

        :param root:
        :param split: 'train', 'val', or 'test'
        :param task_ids: list of task indices (into class_names), e.g. [0] for toxicity
        :param model_name: HuggingFace model id (tokenizer is derived from this)
        :param max_length: max token sequence length
        :param threshold: binarization threshold for continuous labels
        :param max_samples: subsample cap (distributed 80/10/10 across splits)
        :param augmentations: unused, kept for interface compatibility
        """
        self.root = root
        self.split = split
        self.task_ids = task_ids
        self.threshold = threshold
        self.augmentations = augmentations
        self.n_classes = 7

        self.class_names = [
            'toxicity', 'severe_toxicity', 'obscene',
            'threat', 'insult', 'identity_attack', 'sexual_explicit',
        ]

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ── Load data (cached across splits) ──────────────────────────
        csv_path = os.path.join(root, 'train.csv')
        if CivilComments._df_cache is None:
            CivilComments._df_cache = pd.read_csv(csv_path)
        df = CivilComments._df_cache

        # ── Filter to identity-annotated samples only ─────────────────
        # ~18% of samples have identity annotations. Without this filter,
        # unannotated samples (male=NaN) get treated as "not male" by the
        # DEO objective, which corrupts the fairness measurement.
        # ~324k annotated samples remain — plenty for training.
        identity_annotated = df['male'].notna()
        df = df[identity_annotated].reset_index(drop=True)

        # ── Split ─────────────────────────────────────────────────────
        np.random.seed(42)
        n = len(df)
        perm = np.random.permutation(n)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        if 'train' in split:
            indices = perm[:train_end]
        elif 'val' in split:
            indices = perm[train_end:val_end]
        elif 'test' in split:
            indices = perm[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

        if max_samples is not None:
            if 'train' in split:
                split_samples = int(0.8 * max_samples)
            else:
                split_samples = int(0.1 * max_samples)
            if split_samples < len(indices):
                rng = np.random.RandomState(42)
                indices = rng.choice(indices, size=split_samples, replace=False)

        df = df.iloc[indices].copy().reset_index(drop=True)

        # ── Text → tokens ─────────────────────────────────────────────
        texts = df['comment_text'].fillna('').tolist()
        self.encodings = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )

        # ── Labels ────────────────────────────────────────────────────
        # 'target' is the main toxicity score (float 0–1, binarized here).
        # Other columns (severe_toxicity, obscene, threat, insult, etc.)
        # are direct sub-categories of toxicity and highly correlated.
        # For cls+fairness setup, only toxicity (index 0) is used via
        # task_ids=[0, 0].
        label_col_map = {
            'toxicity': 'target',
            'severe_toxicity': 'severe_toxicity',
            'obscene': 'obscene',
            'threat': 'threat',
            'insult': 'insult',
            'identity_attack': 'identity_attack',
            'sexual_explicit': 'sexual_explicit',
        }
        label_matrix = []
        for cls_name in self.class_names:
            col = label_col_map[cls_name]
            vals = df[col].fillna(0).values
            label_matrix.append((vals >= threshold).astype(np.int64))
        self.labels = np.stack(label_matrix, axis=1)  # (N, 7)

        # ── Sensitive attribute ───────────────────────────────────────
        # Since we filtered to annotated samples, NaN here means
        # "annotated but not belonging to this group" → treat as 0.
        self.sensible_attribute = (
            df['male'].fillna(0).values >= threshold
        ).astype(np.float32)

        # Stats for sanity checking
        n_pos = self.sensible_attribute.sum()
        n_toxic = self.labels[:, 0].sum()
        print("Found {} {} samples. Defined tasks: {}".format(
            len(self.labels),
            self.split,
            [self.class_names[i] for i in task_ids] if task_ids else 'all',
        ))
        print("  male={:.1f}% | toxic={:.1f}%".format(
            100 * n_pos / len(self.labels),
            100 * n_toxic / len(self.labels),
        ))

    def __len__(self):
        """__len__"""
        return len(self.labels)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        label = self.labels[index]
        label = torch.from_numpy(label).long()

        input_ids = self.encodings['input_ids'][index]
        attention_mask = self.encodings['attention_mask'][index]

        labels = {'labels_{}'.format(i): label[i] for i in self.task_names()}
        return dict(
            data=input_ids,
            attention_mask=attention_mask,
            sensible_attribute=torch.tensor(self.sensible_attribute[index], dtype=torch.float32),
            **labels,
        )

    def task_names(self):
        return self.task_ids if self.task_ids else range(self.n_classes)


if __name__ == '__main__':
    dst = CivilComments(
        split='val',
        task_ids=[0, 0],
        max_samples=10000,
    )
    bs = 4
    loader = data.DataLoader(dst, batch_size=bs, num_workers=0)

    for i, batch in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  data shape:              {batch['data'].shape}")
        print(f"  attention_mask shape:     {batch['attention_mask'].shape}")
        print(f"  labels_0 (toxicity):     {batch['labels_0']}")
        print(f"  sensible_attribute:      {batch['sensible_attribute']}")
        if i >= 2:
            break