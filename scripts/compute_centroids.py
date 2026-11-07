import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_fashion_mnist
from src.latent.centroids import compute_centroids, save_centroids
from src.models.vae import VAE
from src.utils.config import CENTROID_PATH, CLASS_NAMES, VAE_PATH


def main():
    _, (x_test, y_test) = load_fashion_mnist()
    import tensorflow as tf
    model = tf.keras.models.load_model(VAE_PATH)
    latent = model.encode_mean(x_test).numpy()
    centroids = compute_centroids(latent, y_test, num_classes=10)
    save_centroids(CENTROID_PATH, centroids, CLASS_NAMES)
    print(f"Saved {centroids.shape} centroids to {CENTROID_PATH}")


if __name__ == "__main__":
    main()
