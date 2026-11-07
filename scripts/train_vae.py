import argparse
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_fashion_mnist
from src.models.vae import VAE
from src.utils.config import CLASS_NAMES, LATENT_DIM, MODEL_DIR, SEED, OUTPUT_DIR
from src.utils.visualization import save_reconstruction_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    model = VAE(latent_dim=args.latent_dim, image_shape=(28, 28, 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "fashion_mnist_vae.keras"), save_best_only=False,
    )
    history = model.fit(
        x_train, validation_data=(x_test, x_test), epochs=args.epochs,
        batch_size=args.batch_size, callbacks=[checkpoint], verbose=1,
    )
    np.savez(MODEL_DIR / "training_history.npz", **history.history)
    reconstructions = model.reconstruct(x_test[:10])
    save_reconstruction_grid(
        x_test[:10], reconstructions.numpy(), y_test[:10], CLASS_NAMES,
        OUTPUT_DIR / "reconstructions" / "reconstruction_grid.png",
    )
    print(f"Saved model to {MODEL_DIR / 'fashion_mnist_vae.keras'}")


if __name__ == "__main__":
    main()
