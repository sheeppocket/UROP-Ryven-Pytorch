# UROP visual machine learning

Build a PyTorch model layer by layer in Ryven. Set each layer's size and activation, optionally start with an image encoder, and copy a standalone script to train the assembled model in Colab or on another machine.

## Install and open Ryven

Clone or download this repository. On **x64 Windows**, open [`launchers/windows`](launchers/windows) and double-click `00-install.cmd`. Setup installs Python 3.10 for your Windows account if needed and creates the project's Python environment. Then open `01-blank-canvas.cmd`.

For Git Bash on Windows or x86-64 Linux, follow the [Bash setup instructions](pytorch/README.md#install-and-open-ryven).

## Build a dense classifier

1. In the **Nodes** sidebar, search for **Dense Input** and **Dense Layer**, or find them under **Model Setup**.
2. Set **features** to `2` in **Dense Input**. Connect its **architecture** output to **previous** on the first **Dense Layer**.
3. Set the first layer to **8 neurons** and **ReLU**. Connect its **model** output to **previous** on a second **Dense Layer**. Set the second layer to **2 neurons** and **None** so it returns two raw class scores for cross entropy.

![Ryven graph connecting two input features to an eight-neuron ReLU layer and a two-neuron output layer](docs/images/ryven-dense-layers.png)

## Export and train in Colab

1. Add **Train Model** and connect the last **Dense Layer** to its **model** input.
2. Add **CSV Dataset for Remote Training** and set **csv_path** to `/content/train.csv`. Connect its **dataset** output to **Group Examples into Batches**, then connect **dataloader** to **train_data_loader** on **Train Model**. Ryven uses the CSV path without opening the file.
3. Set **batch_size** to `32` and **shuffle** to **True**. In **Train Model**, select **cross_entropy**, **Adam**, a learning rate of `0.001`, and `10` epochs. Click **Copy Training Code**.

   ![Ryven graph connecting remote CSV batches and the final dense layer to Train Model, with the clipboard confirmation visible](docs/images/ryven-copy-training-code.png)

4. In Colab, upload `train.csv` to `/content`, paste the copied code into a cell, and run it. You can change `DATA_PATH` in the script if the file is somewhere else. The script loads the data on the training machine, uses CUDA when available, reports loss by epoch, and writes `training_result.pth`.

Use a header row, numeric feature columns first, and the target column last. For cross entropy, labels are zero-based integers. This small CSV exercises the two-feature, two-class example above.

```csv
x1,x2,label
-1,-1,0
-1,1,0
1,-1,1
1,1,1
```

The exported script loads the CSV into the training machine's memory. The same script can run as a local Python file with PyTorch installed and `DATA_PATH` set to the file location.

## Testing the layer builder

The first exercise follows the structure of EE4312 Artificial Neural Networks Lab 1 and Project 1. Assemble the model, vary the hidden layers and activations, then compare the results with the data split and training settings held fixed.

### 1. Small classifier or regressor without a backbone

Use the [dense classifier walkthrough](#build-a-dense-classifier) with XOR examples or your own numeric CSV. Compare a single output layer with a hidden layer followed by the output layer. Check that the exported `nn.Linear` dimensions and activations match the graph, then compare held-out accuracy. For scalar regression, use one output neuron, **None** for the final activation, **mse** loss, and a numeric target column. Compare held-out mean squared error.

### 2. U-Net-style VAE on low-resolution images without a backbone

Planned. Build a small convolutional encoder and decoder with a variational bottleneck. Fashion-MNIST, the image dataset from EE4312 Project 1, gives this exercise 28 × 28 grayscale inputs. Test that reconstructions match the input shape, that reconstruction and KL losses are logged separately, and that the decoder can generate samples from a latent vector. Any U-Net skip connections need an explicit design for sampling, when there is no input image to encode.

### 3. Image classifier with a backbone

1. Add **Image Encoder for Code Export** and choose **resnet18**, **resnet34**, or **resnet50**. **DEFAULT** weights with **freeze** set to **True** keeps the pretrained encoder fixed. Ryven stores these settings in the graph. The generated script creates the encoder and fetches the weights on the training machine.
2. Connect the encoder's **architecture** output to **previous** on a **Dense Layer**. Add as many dense layers as needed. For two classes, try **64 neurons, ReLU → 2 neurons, None**. Connect the last layer's **model** output to **Train Model** and use **cross_entropy**.
3. Connect **Image Folder for Remote Training → Group Examples into Batches → Train Model**. Set **image_folder** to a path such as `/content/train` with one subfolder per class. Set the last layer's neuron count to the number of class folders.
4. Click **Copy Training Code** and run it in Colab with the image folder available. The script applies the encoder's preprocessing, loads images in batches, and saves the class names with the weights in `training_result.pth`.

Compare two heads with the encoder and data split held fixed. Check that the generated model emits one score per class and evaluate held-out accuracy. For a frozen run, check that the encoder weights and batch-normalization statistics stay fixed while the head trains.

### 4. Object detector with a backbone

Planned. Expose the encoder's spatial feature maps before global pooling, then build classification and bounding-box heads on top. Start with one labeled object per image to check class and box outputs, coordinate scaling, and overlap with the labeled box on held-out images. Multiple objects per image would also need a way to match predictions to targets and evaluate detections across images.

## Explore the included presets

Open editable lessons for linear classification, XOR, recurrent networks, and image classification with digits or weed photos. The CNN images are bundled. The first use of a pretrained model downloads its weights through torchvision. [Preset launchers and detailed guide](pytorch/README.md#start-with-a-preset)
