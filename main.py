from typing import Optional
from typer import Typer

import os

from train import Trainer
from logger import log

app = Typer(name="AdaIN Style Transfer")


@app.command()
def train(
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
):
    log.info("Setting up training session...")
    trainer = Trainer(
        content_dir,
        style_dir,
        num_iters,
        imsize,
        lr,
        batch_size,
        wt_s,
        ckpt_freq,
        seed,
        ckpt_path,
        device,
    )
    log.info(f"Starting Training session with num_iters={trainer.num_iters}")
    trainer.train()


@app.command()
def infer():
    raise NotImplementedError("Inference not implemented yet!")


@app.command()
def reset(o: bool = False):
    def do_reset():
        log.info("Deleting tensorboard logs and model checkpoints...")
        if os.path.exists("./.temp/"):
            os.remove("./.temp/*")
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
        "+++Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization+++"
    )
    log.info("arxiv: 1703.06868")
    log.info("www.github.com/aadhithya/AdaIN-pytorch")
    app()
