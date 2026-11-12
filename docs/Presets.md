# Presets

[Project overview](../README.md) · [Project Guide](Guide.md)

## Open a graph

After [setting up Ryven](Guide.md#install-and-open-ryven), run these commands from the repository root. On Windows, use Git Bash.

```bash
bash open.sh                                  # Open a blank canvas
bash open.sh pytorch/fashion_mnist_vae.json    # Open the Fashion-MNIST VAE
```

To open another preset, pass its project file to `open.sh`.

| Preset | Ryven project file | What it opens |
|---|---|---|
| Linear classification | [`pytorch/linear_demo.json`](../pytorch/linear_demo.json) | A single dense layer that separates two classes with a straight line. |
| Fashion-MNIST VAE | [`pytorch/fashion_mnist_vae.json`](../pytorch/fashion_mnist_vae.json) | A dense encoder and decoder that learn to reconstruct and generate clothing images from scratch. |
| Five-class digits | [`pytorch/five_class_demo.json`](../pytorch/five_class_demo.json) | A ResNet encoder and dense head for classifying digits 0–4. |
| Weed photos | [`pytorch/weed_demo.json`](../pytorch/weed_demo.json) | A ResNet encoder and dense head for classifying five weed species. |

## Run a preset

- **Linear classification** — Click **Load Examples**, then **Train**. **Check Test Performance** reports accuracy on held-out examples. Click **Copy Training Code** to run the same graph in Colab.
- **Fashion-MNIST VAE** — Click **Copy Training Code** on **Train VAE**. Paste it into a Colab cell and run it. The script downloads Fashion-MNIST where it runs and uses CUDA if available.
- **Five-class digits and weed photos** — Click **Copy Training Code**. Run the script from the repository root to use the bundled training images. In Colab, upload the preset's training folder and update `DATA_PATH`. [Image classifier walkthrough](Guide.md#build-an-image-classifier)

### What the Fashion-MNIST VAE learns

Each 28 × 28 grayscale clothing image goes through a 128-neuron ReLU encoder. **Variational Latent** learns a mean and log variance for each of eight latent values, then samples a vector for the decoder. The decoder uses another 128-neuron ReLU layer and returns 784 pixel logits. Training uses the images as reconstruction targets. [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist)

The script reports reconstruction loss and KL divergence for training and for the held-out test split. Open `fashion_mnist_vae.png` to see the original test images in the top row, their reconstructions in the middle row, and new samples in the bottom row. The trained weights and losses are saved in `fashion_mnist_vae.pth`.

## Save and reopen a graph

In Ryven, choose **File → Save Project** and save the edited graph as a new `.json` file. To reopen it, pass the saved path to `open.sh`. Use quotes if the path contains spaces.

```bash
bash open.sh "saved graphs/my VAE.json"
```
