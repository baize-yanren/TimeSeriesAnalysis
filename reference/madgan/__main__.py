import typer
import madgan

train = madgan.train

_madgan_cli = typer.Typer(name="MAD-GAN CLI")
_madgan_cli.command(name="train")()


@_madgan_cli.callback()
def main() -> None:
    """MAD-GAN Command Line Interface."""

if __name__ == "__main__":
    _madgan_cli()
