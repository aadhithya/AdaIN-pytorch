import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data.sampler import RandomSampler
from torchvision.utils import make_grid
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
from PIL import Image

from model import StyleNet, SRNet
from logger import log
from utils import inv_normz, compute_mean_std
from data import VizDataset, ResizeShortest, ImageDataset


class Trainer:
    def __init__(
        self,
        content_dir: str,
        style_dir: str,
        num_iters: int = 5e3,
        n_epochs: int = 5,
        imsize: int = 128,
        lr: float = 1e-3,
        batch_size: int = 16,
        wt_s: float = 10.0,
        num_samples: int = 1e2,
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
                f"num_iters has a max limit of {self.inf}! Setting num_iters to {self.inf}."
            )
        self.num_iters = min(self.num_iters, self.inf)

        if self.imsize > 512:
            log.warn("Imsize cannot be greater than 512! Setting imsize=512.")

        train_transform = tf.Compose(
            [
                ResizeShortest(self.imsize * 2),
                tf.ToTensor(),
                tf.RandomCrop(self.imsize),
                tf.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        c_ds = ImageDataset(self.content_dir, transform=train_transform)
        s_ds = ImageDataset(self.style_dir, transform=train_transform)

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

        self.ds = VizDataset(
            self.content_dir,
            self.style_dir,
            train_transform,
            n_samples=self.num_samples,
        )
        # self.train_loader = DataLoader(
        #     ds, batch_size=self.batch_size, num_workers=1, drop_last=True
        # )

        self.train_step = 0

        self.model = StyleNet(self.ckpt_path).to(self.device)
        self.optim = Adam(self.model.decoder.parameters(), lr=self.lr)

        log.info(f"PIL.Image.MAX_IMAGE_PIXELS: {Image.MAX_IMAGE_PIXELS}")

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
        # ds = VizDataset(self.content_dir, self.style_dir)
        c_img, s_img = next(
            iter(DataLoader(self.ds, batch_size=8, num_workers=1))
        )
        with torch.no_grad():
            c_img = c_img.float().to(self.device)
            s_img = s_img.float().to(self.device)

            out = self.model(c_img, s_img, return_t=False)
        grid = torch.cat((c_img, s_img, out), 0)
        grid = inv_normz(grid)
        grid = make_grid(grid, nrow=8)
        self.writer.add_image("viz", grid, self.train_step)

    def criterion(self, stylized_img, style_img, t):
        stylized_content_feats = self.model.encoder_forward(
            stylized_img, True
        )
        stylized_feats = self.model.encoder_forward(stylized_img)
        style_feats = self.model.encoder_forward(style_img)

        content_loss = F.mse_loss(t, stylized_content_feats)

        style_loss = 0
        for stz, sty in zip(stylized_feats, style_feats):
            stz_m, stz_s = compute_mean_std(stz)
            sty_m, sty_s = compute_mean_std(sty)
            style_loss += F.mse_loss(stz_m, sty_m) + F.mse_loss(stz_s, sty_s)

        return content_loss + self.wt_s * style_loss

    def train_as_steps(self):
        loop = trange(self.num_iters, desc="Trg Iter: ", dynamic_ncols=True)
        for ix in loop:
            content_img = next(self.content_iter)[0]
            style_img = next(self.style_iter)[0]
            content_img = content_img.float().to(self.device)
            style_img = style_img.float().to(self.device)

            self.optim.zero_grad()

            stylized_img, t = self.model(
                content_img, style_img, return_t=True
            )

            loss = self.criterion(stylized_img, style_img, t)

            loss.backward()

            self.optim.step()
            loop.set_postfix({"Loss": f"{loss.item():.4f}"})
            self.writer.add_scalar("loss", loss.item(), ix)

            if ix % 250 == 0:
                self.viz_samples()

            if (ix + 1) % self.ckpt_freq == 0:
                self.save_model_weights()
            self.train_step += 1

    def train_epoch(self):
        loop = tqdm(
            self.train_loader,
            desc="Trg Iter: ",
            leave=False,
            dynamic_ncols=True,
        )

        for content_img, style_img in loop:
            content_img = content_img.float().to(self.device)
            style_img = style_img.float().to(self.device)

            self.optim.zero_grad()

            stylized_img, t = self.model(
                content_img, style_img, return_t=True
            )

            loss = self.criterion(stylized_img, style_img, t)

            loss.backward()

            self.optim.step()
            loop.set_postfix({"Loss": f"{loss.item():.4f}"})
            self.writer.add_scalar("loss", loss.item(), self.train_step)

            if self.train_step % 50 == 0:
                self.viz_samples()

            self.train_step += 1

    def train(self):
        self.current_ep = 0
        self.train_as_steps()
        if 0:
            for _ in trange(self.n_epochs, desc="Epoch", dynamic_ncols=True):
                self.train_epoch()
                # self.train_as_steps()

                if (self.current_ep + 1) % self.ckpt_freq == 0:
                    self.save_model_weights()

                self.current_ep += 1


class SRTrainer(Trainer):
    def __init__(
        self,
        content_dir: str,
        style_dir: str,
        stylenet_path: str,
        num_iters: int,
        n_epochs: int,
        imsize: int,
        lr: float,
        batch_size: int,
        wt_s: float,
        num_samples: int,
        ckpt_freq: int,
        seed: int,
        ckpt_path: Optional[str],
        device: str,
    ) -> None:
        super().__init__(
            content_dir,
            style_dir,
            num_iters=num_iters,
            n_epochs=n_epochs,
            imsize=imsize,
            lr=lr,
            batch_size=batch_size,
            wt_s=wt_s,
            num_samples=num_samples,
            ckpt_freq=ckpt_freq,
            seed=seed,
            ckpt_path=ckpt_path,
            device=device,
        )

        self.style_net = StyleNet(stylenet_path).to(self.device).eval()
        self.model = SRNet(ckpt_path).to(self.device)

        self.resize = tf.Resize(64)

        self.optim = Adam(self.model.parameters(), lr=lr)

    def save_model_weights(self):
        """
        save_model_weights Saves the weights of the VggDecoder to disk.
        """
        path = f"{self.ckpt_dir}/ckpt-sr-net-{(self.train_step + 1):05d}.tar"
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.transfer_net.state_dict(), path)
        else:
            torch.save(self.model.transfer_net.state_dict(), path)
        return

    def train_as_steps(self):
        self.model.train()
        loop = trange(self.num_iters, desc="Trg Iter: ", dynamic_ncols=True)
        for ix in loop:
            content_img = next(self.content_iter)[0]
            style_img = next(self.style_iter)[0]
            content_img = content_img.float().to(self.device)
            style_img = style_img.float().to(self.device)

            content_img_lr = self.resize(content_img)
            style_img_lr = self.resize(style_img)

            self.optim.zero_grad()

            with torch.no_grad():
                stylized = self.style_net(
                    content_img, style_img, return_t=False
                )

                stylized_lr = self.style_net(
                    content_img_lr, style_img_lr, return_t=False
                )

            stylized_sr = self.model(stylized_lr)

            loss = self.criterion(
                stylized_sr, stylized, content_img, style_img
            )

            loss.backward()

            self.optim.step()
            loop.set_postfix({"Loss": f"{loss.item():.4f}"})
            self.writer.add_scalar("loss", loss.item(), ix)

            if ix % 250 == 0:
                self.viz_samples()

            if (ix + 1) % self.ckpt_freq == 0:
                self.save_model_weights()
            self.train_step += 1

    def criterion(
        self,
        stylized_sr: torch.Tensor,
        stylized: torch.Tensor,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
    ):
        sty_sr_content_feats = self.style_net.encoder_forward(
            stylized_sr, True
        )
        sty_content_feats = self.style_net.encoder_forward(stylized, True)
        content_feats = self.style_net.encoder_forward(content_img, True)

        sr_loss = F.smooth_l1_loss(
            sty_sr_content_feats, sty_content_feats, reduction="mean"
        )

        content_loss = F.smooth_l1_loss(
            sty_sr_content_feats, content_feats, reduction="mean"
        )

        sty_sr_feats = self.style_net.encoder_forward(stylized_sr)
        sty_feats = self.style_net.encoder_forward(style_img)

        style_loss = 0
        for stz, sty in zip(sty_sr_feats, sty_feats):
            stz_m, stz_s = compute_mean_std(stz)
            sty_m, sty_s = compute_mean_std(sty)
            style_loss += F.mse_loss(
                stz_m, sty_m, reduction="mean"
            ) + F.mse_loss(stz_s, sty_s, reduction="mean")

        # style_loss /= len(stylized)

        return (5 * sr_loss) + style_loss + content_loss

    def viz_samples(self):
        # ds = VizDataset(self.content_dir, self.style_dir)
        c_img, s_img = next(
            iter(DataLoader(self.ds, batch_size=8, num_workers=1))
        )
        with torch.no_grad():
            c_img = c_img.float().to(self.device)
            s_img = s_img.float().to(self.device)
            c_lr = self.resize(c_img)
            s_lr = self.resize(s_img)

            out = self.style_net(c_img, s_img, return_t=False)
            out_lr = self.style_net(c_lr, s_lr, return_t=False)
            out_sr = self.model(out_lr)

        grid = torch.cat((c_img, s_img, out, out_sr), 0)
        grid = inv_normz(grid)
        grid = make_grid(grid, nrow=8)
        self.writer.add_image("viz", grid, self.train_step)
