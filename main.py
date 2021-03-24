from typing import Optional
from torch.cuda import device_count
from typer import Typer

import os
from skimage import io
from time import time

from train import Trainer
from logger import log
from infer import run_infer
from export import export_to_onnx

app = Typer(name="AdaIN Style Transfer")


@app.command()
def train(
    content_dir: str,
    style_dir: str,
    num_iters: int = 5e3,
    n_epochs: int = 5,
    imsize: int = 256,
    lr: float = 1e-4,
    batch_size: int = 128,
    wt_s: float = 10.0,
    num_samples: int = 1e2,
    ckpt_freq: int = 500,
    seed: int = 42,
    ckpt_path: Optional[str] = None,
    device: str = "auto",
):
    log.info("Setting up training session...")
    trainer = Trainer(
        content_dir,
        style_dir,
        num_iters,
        n_epochs,
        imsize,
        lr,
        batch_size,
        wt_s,
        num_samples,
        ckpt_freq,
        seed,
        ckpt_path,
        device,
    )
    log.info(f"Starting Training session...🏋🏽‍♂️")
    trainer.train()


@app.command()
def infer(
    content_img: str,
    style_img: str,
    ckpt_dir: str,
    out_path: str = "./outs/output.jpg",
    alpha: float = 1.0,
    imsize: int = 256,
    device="auto",
):
    """
    infer Run Inference.

    Args:
        content_img (str): path to content image.
        style_img (str): path to style image.
        ckpt_dir (str): path to model checkpoint.
        out_path (str): output image save path.
        alpha (float, optional): Style strength [0,1]. Defaults to 1.0.
        imsize (int, optional): Image size to run inference at. Image is resized to imsize before inference and the output is resized to input size using Bilinear interpolation. Defaults to 256.
        device (str, optional): device to run inference on [auto, cpu, cuda]. Defaults to auto.
    """
    st = time()
    out = run_infer(content_img, style_img, ckpt_dir, alpha, imsize, device)
    end = time()
    log.info(f"Inference Successful! Inference Time: {(end-st)} sec.")
    log.info("Saving Image...")
    io.imsave(out_path, out)
    log.info(f"Saving Successful: {out_path}")
    log.info("Done Done London!")


@app.command()
def to_onnx(ckpt_path: str, save_path: str, imsize: int = 1024):
    """
    to_onnx Exports PyTorch Model to ONNX.

    Args:
        ckpt_path (str): Path to model checkpoint.
        save_path (str): Path to save ONNX model to.
        imsize (int, optional): Image size used for model inference. This cannot be changed during onnx inference!. Defaults to 1024.
    """
    export_to_onnx(ckpt_path, save_path, imsize)


@app.command()
def reset(o: bool = False):
    def do_reset():
        log.info("Deleting tensorboard logs and model checkpoints...")
        if os.path.exists("./.temp/"):
            os.system("rm -rf .temp/*")
        log.info("Resetting runid")
        if os.path.exists("./.runid"):
            os.remove(".runid")
        log.info("Reset Complete!!")

    if o:
        do_reset()
    else:
        log.warn("!DANGER DANGER DANGER!")
        res = (
            input(
                "Are you sure you want to reset past logs? [y|n] {default: n} >"
            )
            or "n"
        )
        if res == "y":
            do_reset()
        else:
            log.info("Well, that was a close save! 😅")


if __name__ == "__main__":
    log.info(
        "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
    )
    log.info("arxiv: 1703.06868")
    log.info("www.github.com/aadhithya/AdaIN-pytorch")
    app()
