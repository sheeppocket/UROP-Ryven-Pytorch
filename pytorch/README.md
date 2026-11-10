# PyTorch nodes for Ryven

Import `nodes/nodes.py` in Ryven's node-package dialog. The node picker groups the package into **Data**, **Model Setup**, **Training**, and **Evaluation**. Open **Menu → Windows → Inspector** in the flow and select a node to read what it does.

## Install and open Ryven

Use 64-bit Python 3.10. On Windows, include the Python Launcher if you are setting up Python for the first time.

1. **Windows:** open [`launchers/windows`](../launchers/windows) in File Explorer and double-click `00-install.cmd`. **Git Bash or x86-64 Linux:** from the workspace root, run `bash launchers/bash/00-install.sh`.
2. The installer creates or reuses the workspace's `.venv` and installs the pinned Ryven, Qt, and PyTorch CPU packages. First-time setup needs internet access for the package downloads.
3. Choose a launcher from the table below. Double-click the Windows file, or run `bash launchers/bash/<filename>` from the workspace root. Each launcher starts in the workspace automatically so the graph can find its node package and sample files.

## Start with a preset

Each preset opens directly on an editable graph. You can change its settings, move nodes, or rewire it. Use **File → Save Project** and choose a new filename to keep your own copy. The repository includes the preset data. On a fresh clone, **Load Model** downloads the pretrained weights on first use.

| Lesson | Windows launcher | Bash launcher | What to look for |
|---|---|---|---|
| Blank canvas | `01-blank-canvas.cmd` | `01-blank-canvas.sh` | Start an empty flow with the PyTorch nodes available |
| 1. Linear classification | `02-linear.cmd` | `02-linear.sh` | Two input features, two class scores, and a straight decision boundary |
| 2. Two-layer neural network | `03-neural-network.cmd` | `03-neural-network.sh` | A hidden layer and ReLU let the model separate the XOR quadrants |
| 3A. CNN backbone + classifier | `04-cnn-digits.cmd` | `04-cnn-digits.sh` | A frozen ResNet extracts image features and a new classifier learns five classes |
| 3B. RNN backbone + classifier | `05-rnn.cmd` | `05-rnn.sh` | A recurrent network summarizes an ordered sequence for a two-class classifier |
| Weed photos | `06-weed-photos.cmd` | `06-weed-photos.sh` | The same CNN workflow classifies five species from field photos |

### 1. Linear classification

1. In **Load Sample Data**, leave **Linear boundary** selected and click **Load Examples**. Each point has two features. Class 1 has a positive first feature, so a straight line can separate the two classes.
2. In **Build Sample Model**, leave **Linear classifier** selected and click **Build / Reset Model**. The model is `nn.Linear(2, 2)`. A batch of `B` points enters as a `(B, 2)` tensor and produces `(B, 2)` raw class scores.
3. Click **Train** and watch the loss after each epoch. **Check Held-out Accuracy** automatically evaluates the finished model on 64 separate examples. The training branch has 128 examples and groups them into batches of 16.
4. To see what one epoch does, click **Build / Reset Model**, set `epochs` to 1, and train again. Compare the resulting loss and accuracy with the original run.

Each training batch follows the same PyTorch sequence used in the other lessons:

```python
optimizer.zero_grad()
scores = model(inputs)
loss = torch.nn.functional.cross_entropy(scores, labels)
loss.backward()
optimizer.step()
```

Cross entropy compares the raw scores with the integer class labels. Backpropagation computes gradients, and the optimizer uses those gradients to update the weights. For prediction, the highest score gives the predicted class.

### 2. Two-layer neural network

1. Click **Load Examples** with **XOR quadrants** selected. The same input points as in lesson 1 now have class 1 when their two features have opposite signs. The two class-1 quadrants are diagonally opposite.
2. Click **Build / Reset Model** with **Two-layer neural network** selected, then click **Train**. The model maps two input features to eight hidden activations, applies ReLU, then maps eight activations to two class scores.
3. Keep the XOR data and training settings fixed. Select **Linear classifier**, click **Build / Reset Model**, and train again. Compare the two held-out scores. A single straight boundary cannot separate all four XOR quadrants.

```python
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
```

### 3A. CNN backbone + classifier

Follow the [five-class digits demo](#five-class-demo). Watch **Add Trainable Classifier** map 512 ResNet features to five scores. Compare the spatial dimensions of two convolution outputs for the same image. The backbone stays fixed throughout training.

### 3B. RNN backbone + classifier

1. Click **Load Examples** with **Sequence order** selected. Each example has six time steps and one number per step. Two markers, `+1` and `-1`, appear in opposite orders for the two classes. Paired examples contain the same numbers, so the order carries the class information.
2. Click **Build / Reset Model** with **RNN + classifier** selected, then click **Train**. A batch has shape `(B, 6, 1)`. The RNN reads it one step at a time and produces an eight-value final hidden state per sequence. The classifier converts that state into two class scores.
3. Compare training loss and held-out accuracy with the earlier lessons. Here the backbone and classifier both learn. In lesson 3A, only the classifier learns.

```python
backbone = nn.RNN(input_size=1, hidden_size=8, batch_first=True)
classifier = nn.Linear(8, 2)
_, hidden = backbone(inputs)
scores = classifier(hidden[-1])
```

Use **Build / Reset Model** before comparing training settings in the synthetic lessons. It restores the original initialization. Clicking **Train** again by itself continues from the current weights. The data split and starting weights are fixed in these lessons, and shuffled training batches can still change the outcome. Use the held-out score to check the trained model. For tuning across many experiments, set aside a separate validation split and reserve the test set for the final check.

## Build your own graph

Use the **Blank canvas** launcher to open a new Ryven flow with the PyTorch nodes in the sidebar.

In the **Nodes** sidebar, expand **nodes** and choose a group, or search for a node by name. In **Find Labeled Images**, select **Bundled digits (0–4)** or enter a folder with one subfolder per class. Set `test_folder` if you have a separate test folder. In **Load Vision Model**, select a backbone such as `resnet18`. Add **Convert Images to Tensor**, **Build Dataset from Tensors**, and **Group Examples into Batches** once for training and again for testing. Add **Add Trainable Classifier**, **Train Model**, **Check Test Performance**, and **Predict and View Feature Maps**.

| From | To |
|---|---|
| Vision model `model` | Classifier setup `backbone` and both image converters' `model` inputs. The converters use the preprocessing associated with the selected weights. |
| **Find Labeled Images** `class_names` | Classifier setup `class_names` |
| `train_image_paths` and `test_image_paths` | Each split's **Convert Images to Tensor** input `image_paths` |
| Each split's `image_tensor` and its matching `train_labels` or `test_labels` | **Build Dataset from Tensors** inputs **tensor 1** and **tensor 2**. Click **add input +** to add the label socket. |
| Each split's `dataset` | The matching **Group Examples into Batches** input `dataset`. Turn shuffle on for training. |
| Classifier setup `model` and training batches `dataloader` | **Train Model** inputs `model` and `train_data_loader` |
| Training `trained_model` and test batches `dataloader` | **Check Test Performance** inputs `model` and `test_data_loader` |
| Training `trained_model` and `held_out_image` | **Predict and View Feature Maps** inputs `model` and `image_path` |
| `held_out_label` (optional) | **Predict and View Feature Maps** input `expected_class` to check the single prediction |

Click **Load Model** and **Find Images**, then click **Train**. **Add Trainable Classifier** works with ResNet-18, ResNet-34, ResNet-50, MobileNet V2, VGG-13, and VGG-16. It replaces the final linear classifier with one output per class and keeps the preceding layers fixed in evaluation mode. For ResNet-18 and five classes, the new classifier maps 512 features to five class scores. Use **IMAGENET1K_V1** for the bundled MobileNet V2 weights.

The converter shows the tensor's shape and dtype. Its first dimension counts images, followed by channels, height, and width. The dataset pairs each image with the integer label at the same index. Batching then controls how many of those pairs go into one model step. The two image tensors are held in memory, so use a small subset when working with large custom image collections.

## Five-class demo

Open the **CNN backbone + classifier** preset with `04-cnn-digits`.

1. Click **Load Model** in **Load Pretrained Backbone**.
2. Leave **Bundled digits (0–4)** selected and click **Find Images**. The folders `0` through `4` supply the class names, with 25 training and 10 test images each. Follow the two paths through tensor conversion and dataset construction. The converters show `(125, 3, 224, 224)` for training and `(50, 3, 224, 224)` for testing with the default ResNet-18.
3. Click **Train** in **Train Classifier**. After six epochs, **Check Accuracy on Test Images** reports how many of the test images were classified correctly. Classifier initialization and training batch order can vary between runs.
4. Click **Test Image** in **Predict and Inspect a Test Image** to check the example against its folder label. Click **View Feature Maps** to compare the first and last convolutions. Choose an intermediate layer from either dropdown to compare it with the layer shown in the other panel. To inspect a different image, click **Inspect Image File…**.

The demo data loads locally. The optimizer receives only the new classifier's weights and bias.

The sample PNGs come from the scikit-learn copy of E. Alpaydin and C. Kaynak's Optical Recognition of Handwritten Digits dataset. The sample uses a subset of digits 0–4, converts the pixel values to grayscale, and resizes the images to 96 × 96. [UCI dataset and citation](https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Weed dataset demo

Open the **Weed photos** preset with `06-weed-photos`.

1. Click **Load Model** in **Load Pretrained Backbone**.
2. Click **Find Images**. The two folder paths are already filled in. The node finds 125 training photos and 50 test photos across five classes. Follow each split through tensor conversion, pairing with labels, and batching.
3. Click **Train** in **Train Classifier**. Training fits the classifier head with the pretrained ResNet-18 backbone frozen. **Check Accuracy on Test Images** reports how many of the 50 test photos the classifier gets right.
4. Click **Test Image** in **Predict and Inspect a Test Image** to see a held-out photo and compare its predicted species with its folder label. Click **View Feature Maps** and use the layer dropdowns to compare any two convolution outputs for that photo.

The bundled photos are a [DeepWeeds teaching subset](demo_data/deepweeds/README.md) covering Chinee apple, Lantana, Parkinsonia, Parthenium, and Prickly acacia.

To use your own images, replace `image_folder` and `test_folder` in **Find Labeled Images**. Use one subfolder per class and the same class names in both folders. Leave `test_folder` blank to let Ryven set aside about 20% of each class, with at least one image per split. Automatic splitting requires at least two images per class.

## Copy training code

Code export is available for the image-model workflow. The synthetic lessons above include short PyTorch examples for their models and training steps.

1. Load the model and training images, then click **Copy Training Code** on **Train Model**. The clipboard gets a standalone PyTorch script with the selected model, weights, training split, preprocessing, batch settings, loss, optimizer, and epochs. It starts a fresh training run from the configured starting weights.
2. Paste the script into a Colab cell or a local Python file. The script uses CUDA when available and prints the device, epoch progress, and mean loss. Make the training class folder structure available and point `DATA_ROOT` at it. The script keeps the original training file list. If you used a custom starting checkpoint, make that file available too and update `initial_checkpoint` if needed.
3. Bring `training_result.pth` back to Ryven. For classifier-only training, keep the backbone and class names as configured and enter the file path in **Add Trainable Classifier** → `trained_weights`. Connect that node's `model` output directly to **Check Test Performance** and **Predict and View Feature Maps**. For full-model training, load the checkpoint through **Load Vision Model** with the original architecture.

Folder exports include relative training filenames, and image files and starting checkpoints travel separately. The training script saves only the classifier when it uses a frozen backbone from a torchvision weight preset. Other configurations save the full trained state. The held-out tests stay in Ryven.

The Ryven training and test nodes display successful progress and metrics in their widgets. The copied script prints its training progress in the terminal or notebook cell.

## Usage

- **Load Vision Model** lets you choose an architecture and click **Load Model**. Editing its inputs clears the old model. Empty weights select torchvision's default pretrained weights. Enter `random` or `none` for random initialization. Named weights such as `IMAGENET1K_V1` and paths to matching state dictionaries are supported.
- **Offline weights:** torchvision reuses weights from its download cache. You can also place the matching torchvision `.pth` files in `nodes/pretrained_weights/`. The loader uses the selected weights' preprocessing and category labels for image inference.
- **Convert Images to Tensor** accepts image paths or connected images. With a model connected, it uses the same preprocessing as **Predict and View Feature Maps**. With the model socket empty, `image_size` and `normalize` control conversion.
- **Build Dataset from Tensors** pairs image or feature tensors on **tensor 1** with labels. Use **add input +** to create the label socket. Connected tensors must have equal first dimensions.
- **Labels:** for cross entropy, supply integer class indices with shape `(N,)`. Integer tensors support constant fills; random fills require floating point types. TensorDataset also accepts a list/tuple of tensors on a data socket.
- **Train Model:** connect a model and labelled data loader, select loss/optimizer/rate/epochs, and click **Train**. Input edits never start training. Training runs in a Qt worker; the trained model and sample-weighted epoch mean losses are published on the main thread. Changing inputs during training invalidates its result. Removing the node or closing the app requests cancellation between batches.
- **Check Test Performance** takes a trained model and test batches. The third input defaults to cross entropy and also accepts a supported loss name or callable. Outputs are sample-weighted mean loss and a list of prediction batches. The node displays accuracy for cross entropy with integer class labels. Evaluation restores the model's prior train/eval mode.
- **Predict and View Feature Maps** takes a loaded or trained classifier and one image. Click **Test Image** to see the thumbnail and up to five predictions. Supply an optional `expected_class` as a class name or zero-based index to see whether the top prediction matches. Use **Check Test Performance** for accuracy and loss across a held-out dataset.
- **View Feature Maps** runs the connected image once and shows it beside the first and last convolution outputs. Both the **Reference layer** and **Comparison layer** dropdowns list all captured convolutions in execution order. **Previous** and **Next** step through the comparison layers while keeping the reference fixed. Each panel shows the spatial size and up to eight raw output channels before later operations such as ReLU. Channels are scaled separately around gray (zero).
- **Inspect Image File…** opens the same viewer for a file you choose. It can check a manually entered class. A label supplied by a wire stays with the image on `image_path`, which is the image used by **Test Image** and **View Feature Maps**.
- Text fields accept Python literals (tuples, lists, numbers, strings), with raw text as a fallback. They do not evaluate arbitrary Python expressions.

**Load Vision Model** creates the architecture's standard classifier, normally 1,000 ImageNet classes. A custom checkpoint loaded through that node must match the stock architecture. `layers.py` and the conversion helpers remain experimental.

For headless use, import the package through `ryven.main.packages.nodes_package.import_nodes_package`, set `RYVEN_MODE=no-gui`, and invoke the training node's `train()` method explicitly. Save a trained model with `torch.save(model.state_dict(), path)`; reload into the same architecture. Graph serialization preserves graph configuration, not a trained-model checkpoint.

## Verified environment

Tests were run on Windows with Python 3.10.0, Ryven 3.5.0, ryvencore/ryvencore-qt 0.5.0, PyTorch 2.4.1+cpu, torchvision 0.19.1+cpu, NumPy 1.26.4, Pillow 12.0.0, QtPy 2.4.3, and PySide2 5.15.2.1. This records the existing tested environment; GPU execution has not been verified.

## Run verification

From the workspace root in PowerShell:

```powershell
& .venv/Scripts/python.exe -m unittest discover -s pytorch/tests -v
& .venv/Scripts/python.exe pytorch/tests/gui_smoke.py
& .venv/Scripts/python.exe pytorch/tests/model_smoke.py
```

- Unit/integration tests use real Ryven sessions, ports and connected flows, synthetic labelled data and temporary images/checkpoints. Negative tests intentionally print errors.
- The GUI smoke test uses Qt's offscreen platform. It constructs the node widgets, round-trips widget states, exercises Add Input, Load Model, and the feature-map viewer, trains through the GUI worker while the event loop runs, and checks recovery after training failure.
- The model smoke test requires cached weights for all six architectures. It checks the stock forward pass and a classifier-only training step for each. Cache **IMAGENET1K_V1** for MobileNet V2 and **DEFAULT** for the other architectures before running it. The VGG checks hold the stock model and a frozen copy in memory.

These tests verify functionality, not classifier quality, educational effectiveness, large-dataset performance, or interactive visual layout.
