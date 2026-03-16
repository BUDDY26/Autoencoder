# Autoencoder — FashionMNIST Reconstruction

A fully connected autoencoder implemented in PyTorch, trained on FashionMNIST to learn compact image representations. The model compresses 28×28 grayscale images into a 3-dimensional latent space and reconstructs them, demonstrating unsupervised representation learning on a real image dataset.

**Primary notebook:** `7-2_AE.ipynb`

---

## Architecture

### Encoder
| Layer | Input → Output |
|---|---|
| Linear + ReLU | 784 → 128 |
| Linear + ReLU | 128 → 64 |
| Linear + ReLU | 64 → 12 |
| Linear | 12 → **3** (latent space) |

### Latent Space
The bottleneck is intentionally constrained to **3 dimensions** — small enough to force the model to learn a compact, meaningful representation, while remaining large enough for the network to reconstruct recognizable images.

### Decoder
| Layer | Input → Output |
|---|---|
| Linear + ReLU | 3 → 12 |
| Linear + ReLU | 12 → 64 |
| Linear + ReLU | 64 → 128 |
| Linear + Sigmoid | 128 → **784** |

The decoder mirrors the encoder structure. A Sigmoid activation on the final layer constrains output values to [0, 1], which matches the normalized pixel range of the input images.

---

## Dataset

**FashionMNIST** (via `torchvision.datasets`)
- 70,000 grayscale images at 28×28 pixels
- 10 clothing categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- 60,000 training images, 10,000 test images
- Loaded with `ToTensor()` transform; pixel values normalized to [0, 1]

The dataset will be downloaded automatically to `./data/` on first run if not already present.

---

## Training Setup

| Parameter | Value |
|---|---|
| Loss function | MSELoss (mean squared error between input and reconstruction) |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 128 |
| Epochs | 20 |
| Device | CUDA if available, otherwise CPU |

MSE is used as the reconstruction loss because the task is pixel-level regression — the model must reproduce continuous values in [0, 1] for each pixel.

---

## Outputs

The notebook produces:

1. **Loss curve** — batch-level reconstruction loss plotted over all training iterations, showing convergence over 20 epochs.
2. **Reconstruction comparison** — a grid showing 10 test images (top row: originals, bottom row: reconstructions), visualizing what the model has learned to encode and decode.

---

## How to Run

### Requirements

```bash
pip install -r requirements.txt
```

### Run the notebook

Open `7-2_AE.ipynb` in Jupyter or Google Colab and run all cells. The FashionMNIST dataset will download automatically on first execution.

```bash
jupyter notebook 7-2_AE.ipynb
```

A CUDA-capable GPU is recommended for faster training but is not required — the notebook selects the available device automatically.

---

## Other Notebooks

This repository also contains supplemental coursework notebooks unrelated to the autoencoder:

- `ApliedLLMModule_1Intro.ipynb` — introductory material on the Hugging Face ecosystem and LLM inference
- `Module2Project.ipynb` — prompt injection detection using LoRA fine-tuning on DeBERTa v3

These are not part of the autoencoder project.
