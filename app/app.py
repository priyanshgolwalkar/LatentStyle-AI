import os
import sys
from pathlib import Path
from io import BytesIO

import numpy as np
import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import preprocess_uploaded_image, load_fashion_mnist, save_sample_images
from src.latent.instruction_parser import parse_instruction
from src.latent.centroids import load_centroids
from src.latent.editor import edit_latent, latent_movement
from src.evaluation.clip_evaluator import CLIPEvaluator
from src.utils.config import CLASS_NAMES, CENTROID_PATH, VAE_PATH, OUTPUT_DIR
from src.models.vae import VAE  # Registers the custom model class for loading.


MODEL = None
CENTROIDS = None
STARTUP_STATUS = "Models not loaded."
try:
    import tensorflow as tf
    if VAE_PATH.exists() and CENTROID_PATH.exists():
        MODEL = tf.keras.models.load_model(VAE_PATH)
        CENTROIDS = load_centroids(CENTROID_PATH)
        STARTUP_STATUS = "VAE and latent centroids loaded successfully."
    else:
        STARTUP_STATUS = "Artifacts missing. Run training and centroid scripts before editing."
except Exception as exc:
    STARTUP_STATUS = f"Model loading failed: {exc}"


def _pil_preview(array: np.ndarray) -> Image.Image:
    array = np.asarray(array)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    elif array.ndim == 3 and array.shape[-1] in (3, 4):
        array = Image.fromarray(
            np.clip(array * 255 if array.max() <= 1 else array, 0, 255).astype("uint8")
        ).convert("L")
        array = np.asarray(array)
    return Image.fromarray(np.clip(array * 255 if array.max() <= 1 else array, 0, 255).astype("uint8"), mode="L")


def run_edit(image, instruction: str, alpha: float, enable_clip: bool):
    if image is None:
        raise gr.Error("Please upload a Fashion-MNIST-style grayscale image.")
    if MODEL is None or CENTROIDS is None:
        raise gr.Error(STARTUP_STATUS)
    parsed = parse_instruction(instruction, default_alpha=float(alpha))
    if parsed.source is None or parsed.target is None:
        raise gr.Error(
            "Could not identify both a source and target class. "
            "Example: 'Make this sneaker look more like a sandal.'"
        )
    try:
        uploaded = preprocess_uploaded_image(image)
        mean, _, _ = MODEL.encoder(uploaded, training=False)
        z_original = mean[0].numpy()
        z_edited = edit_latent(
            z_original, CENTROIDS[parsed.source], CENTROIDS[parsed.target], float(alpha)
        )
        reconstruction = MODEL.decoder(mean, training=False).numpy()[0]
        edited = MODEL.decoder(z_edited[None, ...], training=False).numpy()[0]
        movement = latent_movement(z_original, z_edited)

        clip_text = "CLIP optional evaluation mode is disabled."
        clip_target_aligned = False
        evaluator = CLIPEvaluator(enabled=bool(enable_clip))
        if enable_clip:
            try:
                result = evaluator.evaluate(
                    uploaded[0], edited, parsed.source_text, parsed.target_text
                )
                clip_text = (
                    f"Original: target {result.original_target_similarity:.4f} / "
                    f"source {result.original_source_similarity:.4f}<br>"
                    f"Edited: target {result.target_similarity:.4f} / "
                    f"source {result.source_similarity:.4f}<br>"
                    f"More target-aligned after edit: {'Yes' if result.target_aligned else 'No'}"
                )
                clip_target_aligned = result.target_aligned
            except RuntimeError as exc:
                clip_text = f"CLIP unavailable: {exc}"

        edited_path = OUTPUT_DIR / "edits" / "latest_edited.png"
        edited_path.parent.mkdir(parents=True, exist_ok=True)
        _pil_preview(edited).save(edited_path)
        detected = (
            f"Detected source: {parsed.source_text}<br>"
            f"Detected target: {parsed.target_text}<br>"
            f"Latent movement: {movement:.4f}"
        )
        return (
            _pil_preview(image),
            _pil_preview(reconstruction),
            _pil_preview(edited),
            parsed.source_text,
            parsed.target_text,
            clip_text,
            detected,
            str(edited_path),
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Edit failed: {exc}") from exc


with gr.Blocks(title="LatentStyle AI") as demo:
    gr.Markdown("# LatentStyle AI — P38\nText-guided VAE latent-space clothing image editor for Fashion-MNIST.")
    gr.Markdown(f"<b>Status:</b> {STARTUP_STATUS}")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload clothing image")
            sample_dir = OUTPUT_DIR / "samples"
            sample_paths = [str(path) for path in sorted(sample_dir.glob("*.png"))]
            if sample_paths:
                gr.Examples(examples=sample_paths, inputs=image_input, label="Fashion-MNIST test samples")
            instruction = gr.Textbox(
                label="Editing instruction",
                placeholder="Make this sneaker look more like a sandal.",
            )
            alpha = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Editing strength (alpha)")
            enable_clip = gr.Checkbox(label="Enable optional CLIP evaluation", value=False)
            run_button = gr.Button("Edit image", variant="primary")
        with gr.Column():
            original_preview = gr.Image(label="Original")
            reconstruction_preview = gr.Image(label="VAE reconstruction")
            edited_preview = gr.Image(label="Edited image")
    with gr.Row():
        source_output = gr.Textbox(label="Detected source class")
        target_output = gr.Textbox(label="Detected target class")
    clip_output = gr.Markdown(label="CLIP similarity results")
    movement_output = gr.Markdown(label="Latent movement measurement")
    download_output = gr.File(label="Download edited image")

    run_button.click(
        run_edit,
        inputs=[image_input, instruction, alpha, enable_clip],
        outputs=[
            original_preview, reconstruction_preview, edited_preview,
            source_output, target_output, clip_output, movement_output, download_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
