import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data.sampler import RandomSampler
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as tf

import numpy as np
import random

import os
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from fastcore.utils import store_attr
from collections import defaultdict
from typing import Optional, Dict

from model import StyleNet
from logger import log
from utils import inv_normz, img_loader
from data import VizDataset, ResizeShortest


class Trainer:
    def __init__(
        self,
        content_dir: str,
        style_dir: str,
        num_iters: int = 5e3,
        imsize: int = 256,
        lr: float = 1e-4,
        batch_size: int = 128,
        wt_s: float = 10.0,
        ckpt_freq: int = 500,
        seed: int = 42,
        ckpt_path: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        store_attr()
        self.__set_seed()
        self.__resolve_device()
        self.__init_writer()
        self.inf = int(1e32)

        if self.num_iters >= self.inf:
            log.warn(
                "num_iters has a max limit of 1e100! Setting num_iters to 1e100."
            )
        self.num_iters = min(self.num_iters, self.inf)

        if self.imsize > 512:
            log.warn("Imsize cannot be greater than 512! Setting imsize=512.")

        train_transform = tf.Compose(
            [
                ResizeShortest(512),
                tf.ToTensor(),
                tf.RandomCrop(self.imsize),
                tf.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        c_ds = ImageFolder(
            self.content_dir, transform=train_transform, loader=img_loader
        )
        s_ds = ImageFolder(
            self.style_dir, transform=train_transform, loader=img_loader
        )

        c_dl = DataLoader(
            c_ds,
            batch_size=self.batch_size,
            sampler=RandomSampler(
                c_ds, replacement=True, num_samples=self.inf
            ),
            num_workers=2,
        )

        s_dl = DataLoader(
            s_ds,
            batch_size=self.batch_size,
            sampler=RandomSampler(
                s_ds, replacement=True, num_samples=self.inf
            ),
            num_workers=2,
        )

        self.content_iter = iter(c_dl)
        self.style_iter = iter(s_dl)

        self.model = StyleNet(self.ckpt_path).to(self.device)
        self.optim = Adam(self.model.parameters(), lr=self.lr)

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
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(self.seed)
        log.info("Seed Set...🥜")

    def __generate_run_id(self):
        run_id = 0
        if os.path.exists("./.runid"):
            with open("./.runid", "r") as f:
                run_id = int(f.read()) + 1
        with open("./.runid", "w") as f:
            f.write(str(run_id))

        return run_id

    def __init_writer(self):
        run_id = self.__generate_run_id()
        self.log_dir = f"./.temp/{run_id}/"
        self.ckpt_dir = os.path.join(self.log_dir, "ckpt")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

    def save_model_weights(self):
        """
        save_model_weights Saves the weights of the VggDecoder to disk.
        """
        path = (
            f"{self.ckpt_dir}/cktp-style-net-{(self.train_step + 1):05d}.tar"
        )
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.decoder.state_dict(), path)
        else:
            torch.save(self.model.decoder.state_dict(), path)
        return

    def viz_samples(self):
        ds = VizDataset(self.content_dir, self.style_dir)
        c_img, s_img = next(iter(DataLoader(ds, batch_size=8)))
        with torch.no_grad():
            c_img = c_img.float().to(self.device)
            s_img = s_img.float().to(self.device)

            out = self.model(c_img, s_img, return_t=False)
        grid = torch.cat((c_img, s_img, out), 0)
        grid = inv_normz(grid)
        grid = make_grid(grid, nrow=8)
        self.writer.add_image("viz", grid, self.train_step)

    def criterion(self, stylized_img, style_img, t):
        stylized_feats = self.model.encoder_forward(stylized_img)
        style_feats = self.model.encoder_forward(style_img)

        content_loss = F.mse_loss(t, stylized_feats[-1])

        style_loss = 0
        for stz, sty in zip(stylized_feats, style_feats):
            style_loss += F.mse_loss(stz, sty)

        return content_loss + self.wt_s * style_loss

    def train(self):
        self.train_step = 0
        loop = trange(self.num_iters, desc="Trg Iter: ")
        for ix in loop:
            content_img = next(self.content_iter)[0]
            style_img = next(self.style_iter)[0]
            content_img = content_img.float().to(self.device)
            style_img = style_img.float().to(self.device)

            stylized_img, t = self.model(
                content_img, style_img, return_t=True
            )

            loss = self.criterion(stylized_img, style_img, t)

            loss.backward()

            self.optim.step()
            loop.set_postfix({"Loss": f"{loss.item():.4f}"})
            self.writer.add_scalar("loss", loss.item(), ix)

            if ix % 100 == 0:
                self.viz_samples()

            if (ix + 1) % self.ckpt_freq == 0:
                self.save_model_weights()
            self.train_step += 1
