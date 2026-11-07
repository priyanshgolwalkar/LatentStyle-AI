"""Create the comparison model architecture; training is intentionally optional."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.gan import build_lightweight_generator


def main():
    model = build_lightweight_generator()
    print(model.summary())
    print("Lightweight GAN architecture created; train it only with an experimental baseline dataset.")


if __name__ == "__main__":
    main()
