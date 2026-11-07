import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def save_reconstruction_grid(originals: np.ndarray, reconstructions: np.ndarray,
                             labels, class_names, output_path: Path, count: int = 10) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(count, len(originals))
    fig, axes = plt.subplots(2, count, figsize=(1.5 * count, 3))
    for index in range(count):
        axes[0, index].imshow(originals[index, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[0, index].axis("off")
        axes[1, index].imshow(reconstructions[index, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[1, index].axis("off")
        axes[1, index].set_title(class_names[int(labels[index])], fontsize=7)
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstruction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_morph_strip(images: list[np.ndarray], alphas: list[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(images), figsize=(2 * len(images), 2.3))
    if len(images) == 1:
        axes = [axes]
    for ax, image, alpha in zip(axes, images, alphas):
        ax.imshow(np.asarray(image)[..., 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"alpha={alpha:.2f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
