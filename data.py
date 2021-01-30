import numpy as np
import torch
import torchvision.transforms as tf
from torch.utils.data.dataset import Dataset

from skimage import transform, io
from PIL import Image
from fastcore.utils import store_attr
from glob import glob

from logger import log


class ResizeShortest:
    def __init__(self, size=512) -> None:
        assert isinstance(size, (int, tuple))
        self.size = 512

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = np.array(image)
        h, w = image.shape[:2]

        if h > w:
            new_h, new_w = self.size * h / w, self.size
        else:
            new_h, new_w = self.size, self.size * w / h

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img


class VizDataset(Dataset):
    def __init__(
        self,
        content_path: str,
        style_path: str,
        n_samples: int = 8,
    ) -> None:
        super().__init__()

        self.content_path = content_path
        self.style_path = style_path
        self.n_samples = n_samples

        self.__load_paths()

        self.transform = tf.Compose(
            [
                tf.ToTensor(),
                tf.Resize((256, 256)),
                tf.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __load_paths(self):
        content_paths = np.array(glob(f"{self.content_path}/**/*.jpg"))
        style_paths = np.array(glob(f"{self.style_path}/**/*.jpg"))

        max_samples = min(len(content_paths), len(style_paths))

        self.n_samples = min(max_samples, self.n_samples)

        # randomly select n-samples
        self.select_idxs = np.random.permutation(max_samples)[
            : self.n_samples
        ]

        self.content_paths = content_paths[self.select_idxs]
        self.style_paths = style_paths[self.select_idxs]

    def __len__(self):
        return len(self.style_paths)

    def __getitem__(self, index):
        c_path = self.content_paths[index]
        s_path = self.style_paths[index]

        c_img = io.imread(c_path)
        s_img = io.imread(s_path)

        c_img = self.transform(c_img)
        s_img = self.transform(s_img)

        return c_img, s_img
