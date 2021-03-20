import torch
import torchvision.transforms as tf
import numpy as np
from skimage.transform import resize

from PIL import Image
from torchvision.transforms.transforms import Resize

from utils import resolve_device, inv_normz
from logger import log

from model import StyleNet


def load_image(img_path, imsize):
    transform = tf.Compose(
        [
            tf.ToTensor(),
            tf.Resize((imsize, imsize)),
            tf.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    image = np.array(Image.open(img_path).convert("RGB"))
    og_size = image.shape[:-1]

    return (
        transform(image)[
            None,
        ],
        og_size,
    )


def run_infer(
    content_img: str,
    style_img: str,
    ckpt_dir: str,
    alpha: int = 1.0,
    imsize: int = 256,
    device="auto",
):
    log.info("AdaIN")
    device = resolve_device(device)
    log.info(f"running inference on device {device}")
    log.info(f"Loading Content Image: {content_img}")

    content_image, c_size = load_image(content_img, imsize)
    content_image = content_image.float().to(device)
    log.info(f"Loading Style Image: {style_img}")

    style_image, _ = load_image(style_img, imsize)
    style_image = style_image.float().to(device)

    log.info(f"Loading Model: {ckpt_dir}")
    model = StyleNet(ckpt_dir).to(device).eval()

    log.info("Running Inference...")
    with torch.no_grad():
        out, c_f, s_f = model.forward(
            content_image, style_image, alpha, infer=True
        )

    return postprocess(out, c_size), c_f, s_f


def postprocess(img, og_size=None):
    img = img.squeeze().cpu().detach()
    img = inv_normz(img)
    img = img.permute(1, 2, 0)
    img = resize(img, og_size, order=1, preserve_range=True)
    img = (img * 255).astype(np.uint8)
    return img
