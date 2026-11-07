import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.vae import VAE  # Registers the custom model class for loading.
from src.data.dataset import load_fashion_mnist
from src.latent.centroids import load_centroids
from src.latent.editor import edit_latent
from src.utils.config import CENTROID_PATH, CLASS_NAMES, OUTPUT_DIR, VAE_PATH
from src.utils.visualization import save_morph_strip

TRANSITIONS = [
    ("Sneaker", 7, "Sandal", 5),
    ("Shirt", 6, "Coat", 4),
    ("Bag", 8, "Dress", 3),
    ("Ankle boot", 9, "Sneaker", 7),
    ("T-shirt/top", 0, "Pullover", 2),
]
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.00]


def main():
    _, (x_test, y_test) = load_fashion_mnist()
    model = tf.keras.models.load_model(VAE_PATH)
    centroids = load_centroids(CENTROID_PATH)

    for source_name, source_id, target_name, target_id in TRANSITIONS:
        source_indices = np.flatnonzero(y_test == source_id)
        if len(source_indices) == 0:
            raise RuntimeError(f"No test sample available for {source_name}")
        image = x_test[source_indices[0]][None, ...]
        z = model.encode_mean(image).numpy()[0]
        images = []
        for alpha in ALPHAS:
            z_edited = edit_latent(z, centroids[source_id], centroids[target_id], alpha)
            reconstructed = model.decoder(z_edited[None, ...], training=False).numpy()[0]
            images.append(reconstructed)
        name = f"{source_name.replace('/', '-')}_to_{target_name.replace('/', '-')}"
        save_morph_strip(images, ALPHAS, OUTPUT_DIR / "morph_strips" / f"{name}.png")
        print(f"Saved morph: {source_name} -> {target_name}")


if __name__ == "__main__":
    main()
