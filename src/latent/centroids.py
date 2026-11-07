import numpy as np
from pathlib import Path


def compute_centroids(latent_vectors: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    latent_vectors = np.asarray(latent_vectors, dtype="float32")
    labels = np.asarray(labels)
    centroids = np.zeros((num_classes, latent_vectors.shape[1]), dtype="float32")
    for class_id in range(num_classes):
        mask = labels == class_id
        if not np.any(mask):
            raise ValueError(f"No latent vectors found for class {class_id}.")
        centroids[class_id] = latent_vectors[mask].mean(axis=0)
    return centroids


def save_centroids(path: Path, centroids: np.ndarray, class_names=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, centroids=np.asarray(centroids, dtype="float32"), class_names=np.asarray(class_names or []))


def load_centroids(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if "centroids" not in data:
            raise ValueError(f"{path} does not contain 'centroids'.")
        return data["centroids"].astype("float32")
