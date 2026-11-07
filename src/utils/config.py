from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VAE_PATH = MODEL_DIR / "fashion_mnist_vae.keras"
CENTROID_PATH = MODEL_DIR / "class_centroids.npz"
LATENT_DIM = int(os.getenv("LATENT_DIM", "16"))
SEED = int(os.getenv("SEED", "42"))
IMAGE_SHAPE = (28, 28, 1)
CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
