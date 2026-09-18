# Configuration

The main experiment parameters are centralized in:

```text
config.py
```

The configuration used in the experiments is based on:

```python
IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

PRETRAINED = True

FREEZE_BACKBONE = True

USE_CLASS_WEIGHTS = True

EARLY_STOPPING_PATIENCE = 3

SEED = 42
```

All three models use the same experimental configuration, enabling a fairer comparison between architectures.

---

# Training strategy — Transfer Learning

All three models use **transfer learning** starting from ImageNet-pretrained weights.

Two parameters in `config.py` control the behavior:

```python
PRETRAINED = True

FREEZE_BACKBONE = False
```

* **`PRETRAINED = True`** — each architecture is initialized with ImageNet-pretrained weights (`ResNet18_Weights.DEFAULT`, `GoogLeNet_Weights.DEFAULT`, `MobileNet_V3_Small_Weights.DEFAULT`), instead of random weights.
* **`FREEZE_BACKBONE = False`** (current configuration) — the backbone is still initialized with ImageNet weights, but the entire network is trained (**fine-tuning**), also adjusting the pretrained convolutional layers to the Galaxy Zoo domain. See the [Changelog](changelog.en.md) for the reason behind this change.

If `FREEZE_BACKBONE = True`, the feature extractor (backbone) is frozen (`requires_grad = False`), and only the classification head — recreated for Galaxy Zoo's 8 classes — is trained. This technique is known as **feature extraction**, and was this project's original configuration.

The optimizer (`AdamW`) only receives parameters with `requires_grad = True`. With `FREEZE_BACKBONE = False`, this now includes essentially all of the model's parameters (not just the head), increasing training's computational cost — see the [Changelog](changelog.en.md).

The number of trainable parameters (relative to the total) is printed to the console for each trained model, and is also recorded in `metrics.json` (`parameters.total`), allowing the "effective size" of training to be compared across the three models.
