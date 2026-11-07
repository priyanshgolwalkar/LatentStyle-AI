from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image


@dataclass
class CLIPResult:
    source_similarity: float
    target_similarity: float
    original_source_similarity: float
    original_target_similarity: float
    target_aligned: bool
    available: bool
    message: str


class CLIPEvaluator:
    def __init__(self, enabled: bool = False, clip_model_name: str = "ViT-B/32"):
        self.enabled = enabled
        self.clip_model_name = clip_model_name
        self.model = None
        self.preprocess = None

    def load(self) -> None:
        if not self.enabled:
            raise RuntimeError("CLIP is disabled. Enable optional evaluation mode to load it.")
        try:
            import torch
            import clip
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(self.clip_model_name, device=device, jit=False)
            self.device = device
        except Exception as exc:
            raise RuntimeError(
                "CLIP could not be loaded. Optional evaluation is unavailable "
                "because this hardware or environment cannot initialize the model."
            ) from exc

    @staticmethod
    def _as_rgb(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        return np.clip(array * 255 if array.max() <= 1.0 else array, 0, 255).astype("uint8")

    def evaluate(self, original: np.ndarray, edited: np.ndarray,
                 source_name: str, target_name: str) -> Optional[CLIPResult]:
        if not self.enabled:
            return CLIPResult(0, 0, 0, 0, False, False,
                              "CLIP optional evaluation mode is disabled.")
        if self.model is None:
            self.load()
        import torch
        import clip

        original_pil = Image.fromarray(self._as_rgb(original))
        edited_pil = Image.fromarray(self._as_rgb(edited))
        source_text = f"a grayscale clothing image of a {source_name.lower()}"
        target_text = f"a grayscale clothing image of a {target_name.lower()}"
        with torch.no_grad():
            image_input = torch.stack([
                self.preprocess(original_pil),
                self.preprocess(edited_pil),
            ]).to(self.device)
            text_tokens = clip.tokenize([source_text, target_text]).to(self.device)
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).float().cpu().numpy()

        original_source, original_target = similarities[0]
        edited_source, edited_target = similarities[1]
        return CLIPResult(
            source_similarity=float(edited_source),
            target_similarity=float(edited_target),
            original_source_similarity=float(original_source),
            original_target_similarity=float(original_target),
            target_aligned=bool(edited_target > original_target),
            available=True,
            message="CLIP evaluation completed.",
        )
