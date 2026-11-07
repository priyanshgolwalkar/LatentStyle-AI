import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_fashion_mnist, save_sample_images
from src.utils.config import OUTPUT_DIR


def main():
    _, (x_test, _) = load_fashion_mnist()
    paths = save_sample_images(x_test, OUTPUT_DIR / "samples", count=10)
    print(f"Saved {len(paths)} sample images.")


if __name__ == "__main__":
    main()
