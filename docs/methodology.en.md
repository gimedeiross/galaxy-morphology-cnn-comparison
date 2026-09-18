# AI techniques used — theoretical background

This section summarizes the techniques used in the pipeline and **why** each one was chosen for this specific experiment, serving as a draft for the paper's methodology section.

## Transfer Learning

* **ImageNet-pretrained weights** for all three backbones. Avoids training from scratch on a moderately sized dataset (~156k images total) and speeds up convergence, since low/mid-level filters (edges, textures, shapes) learned on ImageNet are reusable for galaxy images.
* **Configurable backbone freezing** (`FREEZE_BACKBONE`). Currently `False`: the whole network is fine-tuned, letting the convolutional features themselves specialize in galaxy morphology — a change made after the frozen backbone capped ResNet18 at ~41% accuracy (see [Changelog](changelog.en.md)). The feature extraction option (`True`, cheaper in time/memory but with a lower performance ceiling) remains available for anyone who wants to compare the trade-off.

## Class balancing

* **Class weights in the loss function** (`nn.CrossEntropyLoss(weight=...)`), computed from each class's frequency in the training split. Necessary because classes like `Irregular` and `Merger` are much rarer in the Galaxy Zoo Dataset than `Round Elliptical` — without this, the model would tend to ignore the minority classes.

## Data Augmentation

* `RandomHorizontalFlip` and `RandomRotation(10)` — galaxies don't have a "correct" orientation; image orientation is an artifact of capture, not a class characteristic.
* `ColorJitter(brightness, contrast)` — simulates exposure variation between astronomical observations.
* Applied **only on the training split**; validation and test use transforms without augmentation, to measure performance under realistic conditions.

## Normalization

* `Normalize` with ImageNet mean/standard deviation — required for transfer learning: the input distribution needs to match what the pretrained weights "expect".

## Regularization

* **Weight decay** (AdamW) — penalizes large weights, reduces overfitting.
* **Early stopping** (`EARLY_STOPPING_PATIENCE`) — stops training when validation macro F1 stops improving, avoiding unnecessary training and late overfitting.
* **Batch Normalization** and **Dropout**, inherited from the Torchvision architectures — not implemented by us, but active and relevant for training stability.

## Auxiliary loss (GoogLeNet)

* The auxiliary classifiers (`aux1`, `aux2`) contribute a weight of `0.3` each to the training loss. An original technique from the Inception/GoogLeNet architecture to inject gradient into intermediate layers and mitigate vanishing gradients in deep networks. Used only during training; validation and testing use exclusively the main output, for a fair evaluation comparable across architectures.

## Optimization

* **AdamW** instead of Adam — decouples weight decay from the adaptive gradient, more theoretically correct than the L2 baked into classic Adam.
* The optimizer only receives parameters with `requires_grad=True`, consistent with backbone freezing — avoids wasting memory/computation on frozen parameters.

## Model selection criterion

* **Validation Macro F1** (not accuracy) as the criterion for saving the best checkpoint and for early stopping. Accuracy can mask poor performance on minority classes; macro F1 weighs every class equally, better aligned with the goal of comparing architectures fairly on an imbalanced dataset.

## Evaluation

* Macro and weighted metrics (precision, recall, F1), per-class `classification_report` and confusion matrix — allow diagnosing *where* each model makes mistakes, not just *how much*.
* Train/validation/test split, with the test set used **exclusively** for the final evaluation, never during training or tuning.

## Reproducibility

* A fixed seed applied to all relevant libraries (`random`, `numpy`, `torch`, `torch.cuda`) and deterministic `cudnn` mode — ensures all three models are compared under exactly the same experimental conditions.

---

# Reproducibility

The project uses a fixed seed:

```python
SEED = 42
```

The seed is applied to the main libraries used in training (`random`, `numpy`, `torch`, `torch.cuda`), along with `cudnn.deterministic = True` and `cudnn.benchmark = False`.

In addition, all three models use the same experimental configuration, allowing for a fairer comparison between architectures.
