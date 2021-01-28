import torch
import numpy as np
import random
from fastcore.utils import store_attr
from typing import Optional, Dict

from model import StyleNet
from logger import log


class Trainer:
    def __init__(
        self,
        ckpt_paths: Optional(Dict[str, str]) = None,
        batch_size: int = 128,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        store_attr()
        self.__set_seed()
        self.__resolve_device()

        self.model = StyleNet()

    def __resolve_device(self):
        self.device = self.device.lower()
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        log.info(f"Using device: {self.device.upper()}.")

    def __set_seed(self):
        torch.backends.cudnn.deterministic = True
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        log.info("Seed Set...🥜")

    def __ckpt(self):
        pass

    def train_epoch(self):
        pass

    def evaluate(self):
        pass

    def train(self):
        pass
