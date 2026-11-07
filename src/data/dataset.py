from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def load_fashion_mnist(normalize: bool = True):
    """Load and normalize Fashion-MNIST. TensorFlow is imported lazily."""
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[..., None].astype("float32")
    x_test = x_test[..., None].astype("float32")
    if normalize:
        x_train /= 255.0
        x_test /= 255.0
    return (x_train, y_train), (x_test, y_test)


def preprocess_uploaded_image(image, target_shape=(28, 28, 1)) -> np.ndarray:
    """Convert an arbitrary uploaded image into a normalized grayscale 28x28 tensor."""
    if image is None:
        raise ValueError("Please upload an image before running an edit.")
    if hasattr(image, "get"):
        pil_image = image if hasattr(image, "convert") else Image.fromarray(np.asarray(image))
    else:
        pil_image = Image.fromarray(np.asarray(image))
    pil_image = ImageOps.grayscale(pil_image).resize((target_shape[0], target_shape[1]), Image.Resampling.LANCZOS)
    array = np.asarray(pil_image, dtype="float32")[..., None] / 255.0
    return np.expand_dims(array, axis=0)


def save_sample_images(x_test: np.ndarray, output_dir: Path, count: int = 10) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(min(count, len(x_test))):
        path = output_dir / f"sample_{index}.png"
        array = np.clip(x_test[index, ..., 0] * 255.0, 0, 255).astype("uint8")
        Image.fromarray(array, mode="L").save(path)
        paths.append(path)
    return paths
