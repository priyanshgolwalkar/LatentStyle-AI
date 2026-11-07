# Model Card: LatentStyle AI VAE

## Intended Use

Academic demonstration of latent-space clothing image editing on Fashion-MNIST. It is not intended for identity-critical or production fashion classification decisions.

## Model

- Convolutional TensorFlow/Keras VAE
- 28x28x1 input
- 16-dimensional latent space by default
- Sigmoid decoder output in [0,1]
- Reconstruction binary cross-entropy plus KL divergence

## Training Data

Fashion-MNIST training split: 60,000 images and labels across ten classes. Evaluation uses the 10,000-image test split.

## Evaluation

Use reconstruction grids, training/validation loss, centroid separation, morph strips, and optional CLIP text-image similarities. Report actual measured values; CLIP is optional because it may be unavailable or domain-mismatched.

## Limitations

See README.md for resolution, grayscale, semantic complexity, disentanglement, centroid, CLIP-domain, real-world, and GAN/VAE limitations.

## Safety and Privacy

Image uploads should be processed ephemerally on the deployed host. Do not upload sensitive imagery. No API key is required for local parsing.
