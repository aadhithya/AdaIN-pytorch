from typing import Callable, Optional
import numpy as np
import torch
import torchvision.transforms as tf
from torch.utils.data.dataset import Dataset
import albumentations.augmentations as A

# from skimage import transform, io
from PIL import Image
from fastcore.utils import store_attr
from glob import glob

from logger import log


class ResizeShortest:
    def __init__(self, size=512) -> None:
        assert isinstance(size, (int, tuple))
        self.size = 512
        self.resize_tf = A.SmallestMaxSize(self.size)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # image = image
        # h, w = image.shape[:2]

        # if h > w:
        #     new_h, new_w = self.size * h / w, self.size
        # else:
        #     new_h, new_w = self.size, self.size * w / h

        # new_h, new_w = int(new_h), int(new_w)

        # resize_tf = tf.Resize((new_h, new_w))

        img = self.resize_tf(image=np.array(image))

        return img["image"]


class VizDataset(Dataset):
    def __init__(
        self,
        content_path: str,
        style_path: str,
        transform: Optional[Callable] = None,
        n_samples: int = 8,
    ) -> None:
        super().__init__()

        self.content_path = content_path
        self.style_path = style_path
        self.n_samples = n_samples

        self.__load_paths()
        if transform is None:
            self.transform = tf.Compose(
                [
                    tf.ToTensor(),
                    tf.Resize((128, 128)),
                    tf.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

    def shuffle_styles(self):
        shfl_idxs = np.random.permutation(len(self.style_paths))
        self.style_paths = self.style_paths[shfl_idxs]

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

        c_img = np.array(Image.open(c_path).convert("RGB"))
        s_img = np.array(Image.open(s_path).convert("RGB"))

        c_img = self.transform(c_img)
        s_img = self.transform(s_img)

        return c_img, s_img
