from PIL import Image
from torch.utils.data import Dataset

class ImgDS(Dataset):
    def __init__(self, paths, ys, tf):
        self.paths = list(paths)
        self.ys = ys
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        y = self.ys[i] if self.ys is not None else -1
        return self.tf(img), y
