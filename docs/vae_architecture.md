# VAE Architecture

## Encoder

- Input: 28 x 28 x 1 grayscale image
- Conv2D 32, stride 2, ReLU
- Conv2D 64, stride 2, ReLU
- Flatten
- Dense 128, ReLU
- Dense latent mean vector
- Dense latent log-variance vector
- Reparameterization sampling: `z = mean + exp(0.5 * log_var) * epsilon`

## Latent Space

- Default latent dimension: 16
- One centroid per Fashion-MNIST class
- Edit: `z_edited = z_source + alpha * (centroid_target - centroid_source)`

## Decoder

- Dense 7 x 7 x 64
- Reshape to 7 x 7 x 64
- Conv2DTranspose 64, stride 2, ReLU
- Conv2DTranspose 32, stride 2, ReLU
- Conv2D 1, sigmoid output

## Loss

- Reconstruction loss: binary cross-entropy summed across image dimensions
- KL loss: negative half of `sum(1 + log_var - mean^2 - exp(log_var))`
- Total: reconstruction loss + beta * KL loss, with beta defaulting to 1.0
