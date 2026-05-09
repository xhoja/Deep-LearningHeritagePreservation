import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

CLASSES = ['intact', 'cracked']  # 0=intact, 1=cracked


def get_transforms(split: str, img_size: int = 224) -> A.Compose:
    if split == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


class HistoricalCrackDataset(Dataset):
    def __init__(self, root: str, split: str = 'train', img_size: int = 224,
                 transform=None, val_ratio: float = 0.15, test_ratio: float = 0.15,
                 random_state: int = 42):
        assert split in ('train', 'val', 'test')
        root = Path(root)

        all_paths, all_labels = [], []
        for label_idx, class_name in enumerate(CLASSES):
            class_dir = root / class_name
            paths = sorted(class_dir.glob('*.jpg')) + sorted(class_dir.glob('*.png'))
            all_paths.extend(paths)
            all_labels.extend([label_idx] * len(paths))

        # stratified split: train / val / test
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_paths, all_labels,
            test_size=val_ratio + test_ratio,
            stratify=all_labels,
            random_state=random_state,
        )
        val_ratio_adj = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=1 - val_ratio_adj,
            stratify=temp_labels,
            random_state=random_state,
        )

        splits = {'train': (train_paths, train_labels),
                  'val':   (val_paths,   val_labels),
                  'test':  (test_paths,  test_labels)}
        self.paths, self.labels = splits[split]
        self.transform = transform or get_transforms(split, img_size)

        # class weights: inverse frequency, useful for sampler & loss
        counts = np.bincount(self.labels)
        self.class_weights = len(self.labels) / (len(CLASSES) * counts)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        return img, self.labels[idx]
