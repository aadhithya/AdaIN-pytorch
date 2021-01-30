import torch
from skimage.io import imread


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
