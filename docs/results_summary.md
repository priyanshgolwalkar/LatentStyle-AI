# P38 Results Summary

## Measured Experiment

All values below are measured; no unexecuted result is included.

| Item | Result |
|---|---:|
| VAE latent dimension | 16 |
| Training epochs | 3 |
| Batch size | 128 |
| Final training loss | 248.395126 |
| Final reconstruction loss | 234.493545 |
| Final KL loss | 13.901587 |
| Final validation loss | 230.025879 |
| Centroid shape | 10 x 16 |
| Automated tests | 15 passed |

## Required Morph Transitions

All transitions contain five rendered images labeled alpha 0.00, 0.25, 0.50, 0.75, and 1.00:

1. Sneaker → Sandal
2. Shirt → Coat
3. Bag → Dress
4. Ankle boot → Sneaker
5. T-shirt/top → Pullover

## Latent Edit Verification

The real Sneaker → Sandal edit measured latent movement 1.6640. The edited image is a model decoder prediction; it is not a fixed image or fabricated output.

## Measured CLIP Evaluation

ViT-B/32 loaded successfully on CPU.

- Original source similarity: 0.2626797259
- Original target similarity: 0.2630435228
- Edited source similarity: 0.2813825607
- Edited target similarity: 0.2845023572
- Target-aligned result: Yes

CLIP scores are affected by Fashion-MNIST's low resolution and grayscale domain mismatch. They should be interpreted as an optional measurement, not proof of a perceptually perfect edit.

## Failure Case

No explicit source and target class in an instruction causes a clear error. Example: arbitrary real-world photos may be mapped correctly to a Fashion-MNIST latent representation but still produce poor or ambiguous edits due to backgrounds, color, viewpoint, and domain shift.
