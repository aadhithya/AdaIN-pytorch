from typer import Typer

from train import Trainer

app = Typer(name="AdaIN Style Transfer")


@app.command("train")
def train():
    pass


@app.command("infer")
def infer():
    raise NotImplementedError("Inference not implemented yet!")
