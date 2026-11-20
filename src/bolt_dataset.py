from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class BoltDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.transform = transform

        self.images = []

        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

        for ext in extensions:
            self.images.extend(
                Path(root_dir).rglob(ext)
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = Image.open(
            self.images[idx]
        ).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, 0