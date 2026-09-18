# Models

## ResNet18

Implemented using the architecture provided by Torchvision:

```text
ResNet18
```

Initialized with ImageNet-pretrained weights (`ResNet18_Weights.DEFAULT`).

The final layer (`fc`) is replaced to produce eight classes; this new layer is always trainable, regardless of `FREEZE_BACKBONE`.

---

## GoogLeNet

Implemented using:

```text
GoogLeNet
```

Initialized with ImageNet-pretrained weights (`GoogLeNet_Weights.DEFAULT`).

The model has two auxiliary classifiers (`aux1` and `aux2`), in addition to the main classifier.

All classifiers (`fc`, `aux1.fc2`, `aux2.fc2`) are adapted to produce eight classes and remain always trainable, even when the rest of the backbone is frozen.

During **training**, the loss function considers:

```text
Loss =
    Main loss
    + 0.3 × Auxiliary loss 1
    + 0.3 × Auxiliary loss 2
```

During **validation and final evaluation**, only the main output is used.

---

## MobileNetV3 Small

Implemented using:

```text
MobileNetV3 Small
```

Initialized with ImageNet-pretrained weights (`MobileNet_V3_Small_Weights.DEFAULT`).

When the backbone is frozen, only `model.features` (the convolutional extractor) has its weights frozen — the entire `model.classifier` (not just the last layer) remains trainable, since MobileNetV3's head is made up of multiple layers (`Linear → Hardswish → Dropout → Linear`) that typically benefit from being trained together.

MobileNetV3 Small was chosen for having a considerably lighter architecture, allowing comparison of not only classification performance but also computational cost and parameter count.

---

# Activation functions per model

What changes between the three architectures in terms of activation functions — in the backbone (inherited from pretrained torchvision) and in the classification head (our own code, in `models/*.py`).

| Model | Backbone activation | New head activation |
|---|---|---|
| **ResNet18** | ReLU (inplace), after each BatchNorm inside the residual blocks. Standard for the original architecture, unchanged. | None — `model.fc` is a single `nn.Linear`. Pure linear classifier over the features. |
| **GoogLeNet** | ReLU inside every Inception module, including the auxiliary branches (`aux1`, `aux2`). Standard for the original architecture, unchanged. | None — `fc`, `aux1.fc2` and `aux2.fc2` are plain `nn.Linear` layers. |
| **MobileNetV3-Small** | Mixed: ReLU in the initial `features` layers (cheaper) and Hardswish in the final layers (more expressive). Design decision from the original paper, unchanged. | Hardswish — inherited from the original `classifier` (`Linear -> Hardswish -> Dropout -> Linear`); only the last `Linear` layer is replaced, so the intermediate Hardswish stays active. |

**Points worth highlighting:**

* No activation function is added or chosen by us — they all come from within torchvision's pretrained backbones. The only code-level decision about activations is **not** adding any to the new ResNet and GoogLeNet heads, versus **inheriting** the Hardswish already present in MobileNet's head.
* This creates an asymmetry between the three models: MobileNet has a "factory" non-linear head, while ResNet and GoogLeNet end up with pure *linear probing* (common in transfer learning with a frozen backbone). This difference is worth mentioning in the paper's methodology, since it can influence the comparison — it's not just the backbone that differs between models, head capacity differs too.
* The comments equivalent to this table are in the docstrings of `create_model()` in `models/resnet.py`, `models/googlenet.py` and `models/mobilenet.py`.
