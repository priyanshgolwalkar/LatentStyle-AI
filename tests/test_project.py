import tempfile
from pathlib import Path
import numpy as np
import pytest


def test_parser_supported_transition():
    from src.latent.instruction_parser import parse_instruction
    parsed = parse_instruction("Make this sneaker look more like a sandal.", 1.0)
    assert parsed.source == 7
    assert parsed.target == 5
    assert parsed.alpha == 1.0


@pytest.mark.parametrize(
    "instruction,source,target",
    [
        ("Change this shirt into a coat.", 6, 4),
        ("Make this ankle boot resemble a sneaker.", 9, 7),
    ],
)
def test_parser_more_transitions(instruction, source, target):
    from src.latent.instruction_parser import parse_instruction
    parsed = parse_instruction(instruction)
    assert (parsed.source, parsed.target) == (source, target)


@pytest.mark.parametrize("instruction,source,target", [
    ("Make this T-shirt look more like a pullover.", 0, 2),
    ("Change this trouser into a dress.", 1, 3),
    ("Make this bag resemble a coat.", 8, 4),
])
def test_parser_aliases_and_transitions(instruction, source, target):
    from src.latent.instruction_parser import parse_instruction
    parsed = parse_instruction(instruction, 0.5)
    assert (parsed.source, parsed.target) == (source, target)
    assert parsed.alpha == 0.5


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_alpha_editor_values(alpha):
    import numpy as np
    from src.latent.editor import edit_latent
    z = np.zeros(16, dtype="float32")
    source = np.zeros(16, dtype="float32")
    target = np.ones(16, dtype="float32")
    edited = edit_latent(z, source, target, alpha)
    assert np.linalg.norm(edited) == pytest.approx(alpha * 4.0)


def test_all_centroids_and_required_directions():
    from pathlib import Path
    import numpy as np
    centroid_path = Path(__file__).resolve().parents[1] / "models" / "class_centroids.npz"
    if not centroid_path.exists():
        pytest.skip("Trained centroid artifact is not available.")
    centroids = np.load(centroid_path)["centroids"]
    assert centroids.shape == (10, 16)
    assert np.isfinite(centroids).all()
    for source, target in [(7, 5), (6, 4), (8, 3), (9, 7), (0, 2)]:
        assert np.linalg.norm(centroids[target] - centroids[source]) > 0.0


def test_latent_edit_changes_vector():
    from src.latent.editor import edit_latent, latent_movement
    z = np.zeros(16, dtype="float32")
    source = np.zeros(16, dtype="float32")
    target = np.ones(16, dtype="float32")
    edited = edit_latent(z, source, target, alpha=1.0)
    assert edited.shape == (16,)
    assert latent_movement(z, edited) > 0


def test_centroids_and_save_load():
    from src.latent.centroids import compute_centroids, save_centroids, load_centroids
    vectors = np.vstack([np.zeros((3, 4)), np.ones((3, 4))])
    labels = np.array([0, 0, 0, 1, 1, 1])
    centroids = compute_centroids(vectors, labels, num_classes=2)
    assert np.allclose(centroids[0], np.zeros(4))
    assert np.allclose(centroids[1], np.ones(4))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "centroids.npz"
        save_centroids(path, centroids, ["zero", "one"])
        loaded = load_centroids(path)
        assert np.allclose(loaded, centroids)


def test_preprocessing_shape():
    from src.data.dataset import preprocess_uploaded_image
    image = np.zeros((64, 64, 3), dtype="uint8")
    batch = preprocess_uploaded_image(image)
    assert batch.shape == (1, 28, 28, 1)
    assert batch.min() == 0
    assert batch.max() == 0


def test_vae_forward_and_reconstruction_shape():
    import tensorflow as tf
    from src.models.vae import VAE
    model = VAE(latent_dim=16)
    images = tf.zeros((2, 28, 28, 1))
    reconstruction = model.reconstruct(images)
    latent_mean, latent_log_var, latent_z = model.encoder(images, training=False)
    assert reconstruction.shape == (2, 28, 28, 1)
    assert latent_mean.shape == latent_log_var.shape == latent_z.shape == (2, 16)


def test_morph_strip_generation():
    from src.utils.visualization import save_morph_strip
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "strip.png"
        images = [np.zeros((28, 28, 1)) for _ in range(5)]
        save_morph_strip(images, [0, .25, .5, .75, 1], path)
        assert path.exists()


def test_clip_disabled_is_explicit():
    from src.evaluation.clip_evaluator import CLIPEvaluator
    result = CLIPEvaluator(enabled=False).evaluate(
        np.zeros((28, 28, 1)), np.zeros((28, 28, 1)), "Sneaker", "Sandal"
    )
    assert result.available is False
    assert result.source_similarity == 0
    assert "disabled" in result.message.lower()


def test_real_sneaker_to_sandal_edit():
    from pathlib import Path
    import tensorflow as tf
    from src.latent.centroids import load_centroids
    from src.latent.editor import edit_latent, latent_movement

    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / "fashion_mnist_vae.keras"
    centroid_path = root / "models" / "class_centroids.npz"
    if not model_path.exists() or not centroid_path.exists():
        pytest.skip("Trained model artifacts are not available.")
    model = tf.keras.models.load_model(model_path)
    centroids = load_centroids(centroid_path)
    z = np.zeros((1, 28, 28, 1), dtype="float32")
    z_latent = model.encode_mean(z).numpy()[0]
    edited = edit_latent(z_latent, centroids[7], centroids[5], 1.0)
    assert latent_movement(z_latent, edited) > 0
    decoded = model.decoder(edited[None, ...], training=False).numpy()
    assert decoded.shape == (1, 28, 28, 1)


def test_gradio_application_imports():
    from app.app import demo, MODEL, CENTROIDS
    assert demo is not None
    assert MODEL is not None
    assert CENTROIDS is not None
