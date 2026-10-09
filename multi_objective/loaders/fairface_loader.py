import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data


class FairFace(data.Dataset):

    race_classes   = ['White', 'Black', 'Latino_Hispanic', 'East Asian',
                      'Southeast Asian', 'Indian', 'Middle Eastern']
    gender_classes = ['Male', 'Female']
    age_classes    = ['0-2', '3-9', '10-19', '20-29', '30-39',
                      '40-49', '50-59', '60-69', 'more than 70']

    race_mapping   = {c: i for i, c in enumerate(race_classes)}
    gender_mapping = {c: i for i, c in enumerate(gender_classes)}
    age_mapping    = {c: i for i, c in enumerate(age_classes)}

    TEST_FRACTION = 0.20
    SEED = 42

    def __init__(self, split, task_ids=[2], sensitive_attr='race',
                 root='data/fairface', dim=(3, 224, 224),
                 augmentations=None, **kwargs):
        """
        :param split:          'train', 'val', or 'test'
        :param task_ids:       subset of [0, 1, 2] — 0=race, 1=gender, 2=age.
                               Empty list returns all three.
        :param sensitive_attr: attribute used as fairness grouping signal.
                               One of 'race', 'gender', 'age'. Always returned
                               as batch['sensitive'] regardless of task_ids.
        :param root:           path to the FairFace root directory
        :param dim:            (C, H, W) — H and W must be equal
        :param augmentations:  optional callable applied to a uint8 numpy image
                               before the standard torchvision transform
        """
        assert split in ('train', 'val', 'test'), "split must be 'train', 'val', or 'test'"
        assert dim[-1] == dim[-2], "Height and width must be equal"
        assert sensitive_attr in ('race', 'gender', 'age'), (
            "sensitive_attr must be one of 'race', 'gender', 'age'"
        )

        self.root           = root
        self.split          = split
        self.task_ids       = task_ids
        self.augmentations  = augmentations
        self.n_classes      = 3
        self.class_names    = ['race', 'gender', 'age']
        self.sensitive_attr = sensitive_attr
        self.sensitive_idx  = self.class_names.index(sensitive_attr)  # 0 for race

        self.transform = transforms.Compose([
            transforms.Resize(dim[-1]),
            transforms.CenterCrop(dim[-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        df = self._load_and_split(root, split)

        self.file_list = [os.path.join(root, p) for p in df['file'].tolist()]
        self.label_list = [
            [
                self.race_mapping[row['race']],
                self.gender_mapping[row['gender']],
                self.age_mapping[row['age']],
            ]
            for _, row in df.iterrows()
        ]

        if len(self.file_list) < 2:
            raise Exception(f"No files for split=[{split}] found in {root}")

        print("Found {} {} images. Defined tasks: {}. Sensitive attribute: {}".format(
            len(self.file_list),
            self.split,
            [self.class_names[i] for i in self.task_names()],
            self.sensitive_attr,
        ))

    def _load_and_split(self, root: str, split: str) -> pd.DataFrame:
        if split == 'val':
            return pd.read_csv(os.path.join(root, 'fairface_label_val.csv'))

        df = pd.read_csv(os.path.join(root, 'fairface_label_train.csv'))
        df = df.sample(frac=1, random_state=self.SEED).reset_index(drop=True)

        n_test = int(len(df) * self.TEST_FRACTION)

        if split == 'test':
            return df.iloc[:n_test]
        else:
            return df.iloc[n_test:]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        label = torch.tensor(self.label_list[index]).long()

        img = Image.open(self.file_list[index]).convert('RGB')

        if self.augmentations is not None:
            img = self.augmentations(np.array(img, dtype=np.uint8))

        img = self.transform(img)

        # Task labels — only the requested prediction targets
        labels = {f'labels_{i}': label[i] for i in self.task_names()}

        # Sensitive attribute — always included, separate from task labels
        labels['sensible_attribute'] = label[self.sensitive_idx]

        return dict(data=img, **labels)

    def task_names(self):
        return self.task_ids if self.task_ids else range(self.n_classes)

    def decode_labels(self, label_tensor):
        return {
            'race':   self.race_classes[label_tensor[0]],
            'gender': self.gender_classes[label_tensor[1]],
            'age':    self.age_classes[label_tensor[2]],
        }

    @property
    def n_sensitive_groups(self):
        """Number of groups for the active sensitive attribute."""
        return {
            'race':   len(self.race_classes),    
            'gender': len(self.gender_classes), 
            'age':    len(self.age_classes),     
        }[self.sensitive_attr]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for s in ('train', 'val', 'test'):
        d = FairFace(split=s, task_ids=[2], sensitive_attr='race')
        print(f"  {s}: {len(d)} samples, sensitive groups: {d.n_sensitive_groups}")

    dst = FairFace(split='val', task_ids=[2], sensitive_attr='race')
    loader = data.DataLoader(dst, batch_size=4, num_workers=0, shuffle=True)

    for batch in loader:
        print("Keys:     ", list(batch.keys()))
        print("data:     ", batch['data'].shape)
        print("labels_2: ", batch['labels_2'])   # age indices
        print("sensible_attribute:", batch['sensible_attribute'])   # race indices
        break