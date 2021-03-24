import torch
from skimage.io import imread
from logger import log
import numpy as np


def compute_mean_std(
    feats: torch.Tensor, eps=1e-8, infer=False
) -> torch.Tensor:
    assert (
        len(feats.shape) == 4
    ), "feature map should be 4-dimensional of the form N,C,H,W!"
    #  * Doing this to support ONNX.js inference.
    if infer:
        n = 1
        c = 512  # * fixed for vgg19
    else:
        n, c, _, _ = feats.shape

    feats = feats.view(n, c, -1)
    mean = torch.mean(feats, dim=-1).view(n, c, 1, 1)
    std = torch.std(feats, dim=-1).view(n, c, 1, 1) + eps

    return mean, std


def inv_normz(img):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(img.device)
    mean = (
        torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(img.device)
    )
    out = torch.clamp(img * std + mean, 0, 1)
    return out


def normz(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    if isinstance(img, np.ndarray):
        img = torch.tensor(img)

    mean = torch.tensor(mean).to(img.device)
    std = torch.tensor(std).to(img.device)

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    return (img - mean) / std


def img_loader(path: str):
    img = imread(path)
    return img


def resolve_device(device: str = "auto"):
    if device.lower() in ["auto", "cpu", "cuda"]:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return device
    else:
        log.warn(
            f"{device} should be one of [auto, cpu, cuda]! Defaulting to cpu."
        )
        return "cpu"
