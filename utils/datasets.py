import random

from collections import defaultdict
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms
        self.label_to_indices = self._create_label_indices()

    def _create_label_indices(self):
        label_to_indices = defaultdict(lambda: [])
        for idx, label in enumerate(self.dataset['label']):
            label_to_indices[label].append(idx)

        return label_to_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        anchor, label = self.dataset[index].values()

        positive_index = index
        count = 0
        while positive_index == index and count < 20:
            positive_index = random.choice(self.label_to_indices[label])
            count += 1
        positive, _ = self.dataset[positive_index].values()

        negative_label = random.choice(list(set(self.label_to_indices.keys()) - set([label])))
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative, _ = self.dataset[negative_index].values()

        anchor = anchor.convert("RGB")
        positive = positive.convert("RGB")
        negative = negative.convert("RGB")

        if self.transforms:
            anchor = self.transforms(anchor)
            positive = self.transforms(positive)
            negative = self.transforms(negative)

        return anchor, positive, negative
    
class CustomTestDataset(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        origin_img, _ = self.dataset[idx].values()

        origin_img = origin_img.convert("RGB")

        if self.transforms:
            transform_img = self.transforms(origin_img)

        return transform_img