# Academic Report Support

## U1 — Class Centroids

Fashion-MNIST images were normalized to [0,1], reshaped as 28x28x1, encoded with the trained VAE, and mean-averaged by class label. The resulting artifact has shape 10 x 16 and was saved to `models/class_centroids.npz`.

## U2 — Natural-Language Editing

The application uses a deterministic local parser for source and target aliases. Optional API integration is not required. The local result is mapped to centroid subtraction and decoded with an alpha-scaled latent direction. The included GAN baseline is architectural rather than an experimentally trained competitor.

## U3 — CLIP Evaluation

ViT-B/32 was loaded on CPU and cosine similarities were measured for original and edited grayscale images against source and target text descriptions. The measured result is domain-sensitive and was reported honestly rather than treated as definitive semantic proof.

## U4 — Training and Morphs

The TensorFlow/Keras VAE uses reparameterization sampling, reconstruction loss, KL divergence, validation tracking, checkpoint saving, and decoder reconstruction. Five morph PNGs were generated at alphas 0, 0.25, 0.50, 0.75, and 1.

## U5 — Critical Analysis

Fashion-MNIST is low-resolution, grayscale, and semantically limited. Latent disentanglement is imperfect, centroid directions are average vectors, arbitrary photographs are out of distribution, CLIP is natural-image-oriented, and generated images differ from real clothing. These limitations should be discussed separately from measured model behavior.
