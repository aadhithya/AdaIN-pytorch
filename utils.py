import torch
from skimage.io import imread


def compute_mean_std(feats: torch.Tensor, eps=1e-8) -> torch.Tensor:
    assert (
        len(feats.shape) == 4
    ), "feature map should be 4-dimensional of the form N,C,H,W!"
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


def img_loader(path: str):
    img = imread(path)
    return img
