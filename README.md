# UROP visual machine learning

Build a PyTorch model layer by layer in Ryven. Choose layer widths and activations, optionally add a ResNet encoder, then copy a standalone training script to run in Colab or another Python environment.

![Ryven canvas with two input features, an eight-neuron ReLU layer, and a two-neuron output layer](docs/images/ryven-dense-layers.png)

## Quick start

1. Clone or download this repository and run `bash setup.sh` from its root. Use **Git Bash** on Windows. On Linux, install Python 3.10 with `venv` support first.
2. Run `bash open.sh pytorch/fashion_mnist_vae.json` to open the Fashion-MNIST VAE preset.
3. Click **Copy Training Code** on **Train VAE**, then paste it into a Colab cell and run it. The script trains the VAE and saves a preview of original images, reconstructions, and generated samples.

See the [setup instructions](docs/Guide.md#install-and-open-ryven) for supported platforms and [Presets](docs/Presets.md) for the other project files.

## Documentation

| Guide | What you'll find |
|---|---|
| [Project Guide](docs/Guide.md) | Setup, screenshot walkthroughs, architecture, and how to extend the node package. |
| [Presets](docs/Presets.md) | Included Ryven project files and how to open, run, and save them. |
