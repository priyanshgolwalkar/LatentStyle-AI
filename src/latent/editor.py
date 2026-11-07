import numpy as np


def edit_latent(z_source: np.ndarray, centroid_source: np.ndarray,
                centroid_target: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    z_source = np.asarray(z_source, dtype="float32")
    if z_source.shape[-1] != centroid_source.shape[-1] or centroid_source.shape != centroid_target.shape:
        raise ValueError("Latent vector and centroid dimensions do not match.")
    if not 0.0 <= alpha <= 2.0:
        raise ValueError("Alpha must be between 0 and 2.")
    direction = np.asarray(centroid_target - centroid_source, dtype="float32")
    return z_source + alpha * direction


def latent_movement(z_original: np.ndarray, z_edited: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(z_edited) - np.asarray(z_original)))
