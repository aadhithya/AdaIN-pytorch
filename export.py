import torch

from logger import log
from utils import resolve_device
from model import StyleNet


def export_to_onnx(ckpt_dir: str, out_path: str, imsize=1024):
    log.info("PyTorch --> ONNX Exporter.")
    log.info(f"Loading model with ckpt: {ckpt_dir}")
    device = resolve_device()
    model = StyleNet(ckpt_dir).to(device)

    input_names = ["content_img", "style_img", "alpha"]
    output_names = ["output_img"]

    return_t = torch.Tensor([0]).to(device)
    infer = torch.Tensor([1]).to(device)
    alpha = torch.Tensor([1.0]).to(device)

    dummy_tensor = torch.ones(1, 3, imsize, imsize).to(device)
    dummy_input = (dummy_tensor, dummy_tensor, alpha, return_t, infer)

    log.info("Exporting model to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=input_names,
        output_names=output_names,
        verbose=True,
        opset_version=9,
    )
    log.info(
        f"Model with input size {imsize} exported to ONNX at {out_path}."
    )
    log.info("Done Done London!")
