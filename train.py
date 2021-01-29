import torch
from torch.optim import Adam
import torch.nn.functional as F
from torchvision.utils import make_grid

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
from utils import inv_normz


class Trainer:
    def __init__(
        self,
        epochs: int,
        lr: float = 1e-3,
        batch_size: int = 128,
        wt_s: float = 10.0,
        ckpt_freq: int = 10,
        seed: int = 42,
        ckpt_path: Optional(str) = None,
        device: str = "auto",
    ) -> None:
        store_attr()
        self.__set_seed()
        self.__resolve_device()
        self.__init_writer()

        self.model = StyleNet()
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
        torch.backends.cudnn.deterministic = True
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
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
            f"{self.ckpt_dir}/cktp-style-net-{(self.current_ep + 1):03d}.tar"
        )
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.decoder.state_dict(), path)
        else:
            torch.save(self.model.decoder.state_dict(), path)
        return

    def viz(self, content_img, style_img, stylized_img, n=8, train=True):
        content_img = content_img[:n]
        style_img = style_img[:n]
        stylized_img = stylized_img[:n]
        grid = make_grid(
            torch.cat((content_img, style_img, stylized_img), 0), nrow=n
        )

        tag = "train" if train else "val"

        self.writer.add_image(tag, grid, self.current_ep)

    def criterion(self, stylized_img, style_img, t):
        stylized_feats = self.model.encoder_forward(stylized_img)
        style_feats = self.model.encoder_forward(style_img)

        content_loss = F.mse_loss(t, stylized_feats[-1])

        style_loss = 0
        for stz, sty in zip(stylized_feats, style_feats):
            style_loss += F.mse_loss(stz, sty)

        return content_loss + self.wt_s * style_loss

    def train_epoch(self):
        self.metric = defaultdict(list)
        loop = tqdm(self.train_loader, desc="Trg Itr: ", leave=False)
        for content_img, style_img in loop:
            content_img = content_img.float().to(self.device)
            style_img = style_img.float().to(self.device)

            stylized_img, t = self.model(
                content_img, style_img, return_t=True
            )

            loss = self.criterion(stylized_img, style_img, t)

            loss.backward()

            self.optim.step()
            self.metric["loss"] += [loss.item()]
            self.writer.add_scalar(
                "step/train", loss.item(), global_step=self.train_step
            )
            self.train_step += 1

        self.viz(content_img, style_img, stylized_img)
        return sum(self.metric["loss"]) / len(self.metric["loss"])

    def evaluate(self):
        loop = tqdm(self.val_loader, desc="Val Itr: ", leave=False)
        self.metric = defaultdict(list)
        with torch.no_grad():
            for content_img, style_img in loop:
                content_img = content_img.float().to(self.device)
                style_img = style_img.float().to(self.device)

                stylized_img, t = self.model(
                    content_img, style_img, return_t=True
                )

                loss = self.criterion(stylized_img, style_img, t)

                self.metric["loss"] += [loss.item()]
                self.writer.add_scalar(
                    "step/val", loss.item(), global_step=self.train_step
                )
                self.val_step += 1

        self.viz(content_img, style_img, stylized_img, train=False)
        return sum(self.metric["loss"]) / len(self.metric["loss"])

    def train(self):
        self.train_step = 0
        self.val_step = 0
        self.current_ep = 0
        loop = trange(self.epochs, desc="Epoch:")
        for _ in loop:
            loss = self.train_epoch()
            loop.set_postfix({"TrgLoss": f"{loss:0.4f}"})
            loss = self.evaluate()
            loop.set_postfix({"ValLoss": f"{loss:0.4f}"})

            if (self.current_ep + 1) % self.ckpt_freq == 0:
                self.save_model_weights()

            self.current_ep += 1
