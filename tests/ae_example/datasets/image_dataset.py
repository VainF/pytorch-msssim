from torch.utils import data 
import os
from PIL import Image

class ImageDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = list(os.listdir(root))
        self.images.sort()
    def __getitem__(self, idx):
        img = Image.open(os.path.join( self.root, self.images[idx] ))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

