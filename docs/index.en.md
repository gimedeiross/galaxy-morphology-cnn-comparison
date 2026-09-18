# Galaxy Zoo — CNN Architecture Comparison

Project for the **Artificial Intelligence II** course: training, evaluating and comparing Convolutional Neural Network architectures applied to galaxy classification.

The project uses the **`mrJordi0/galaxy-zoo-dataset`** dataset, available on Hugging Face, and compares three architectures:

* **ResNet18**
* **GoogLeNet**
* **MobileNetV3 Small**

All three models use **transfer learning**: they start from ImageNet-pretrained weights and, currently, train the whole network (full fine-tuning — see [Changelog](changelog.en.md)). The project also supports the cheaper variant of training only the classification head (frozen backbone), configurable in `config.py`.

The goal is to run the experiments under controlled conditions and collect metrics that can be used in writing the scientific paper.

## Project goal

The goal is not simply to identify which architecture achieves the highest accuracy. The analysis aims to compare the architectures considering:

* classification performance;
* Macro F1;
* Weighted F1;
* per-class performance;
* confusion matrix;
* training behavior;
* total and trainable parameter counts (relevant with a frozen backbone);
* training time;
* evaluation time;
* GPU memory usage.

This way, the results can be used to discuss the **performance vs. computational cost trade-offs** of the architectures evaluated in the Artificial Intelligence II scientific paper.

## Technologies

| Library | Why it's used in this project |
|---|---|
| **PyTorch** | Deep learning framework used for all three models, training, loss and optimizer. Standard choice for working with `torchvision` and having explicit control over the training loop (useful here because GoogLeNet needs special handling of its auxiliary outputs — see `train.py`). |
| **Torchvision** | Provides the three architectures (`resnet18`, `googlenet`, `mobilenet_v3_small`) already with ImageNet-pretrained weights (`*_Weights.DEFAULT`), avoiding reimplementing the networks and enabling direct transfer learning. |
| **Hugging Face Datasets** | Loads the `mrJordi0/galaxy-zoo-dataset` dataset directly from the Hub (`load_dataset`), with automatic local caching and `train_test_split` support to generate the validation split. Preferred over manually downloading/organizing files into folders (as `torchvision.datasets.ImageFolder` would require), since the dataset is already distributed in this format. |
| **NumPy** | Numerical support used indirectly by `torch`, `sklearn` and `matplotlib` (e.g. `np.arange` for the confusion matrix ticks in `evaluate.py`). |
| **Pillow** | Handles the images loaded by `datasets` (`.convert("RGB")` in `dataset.py`) before applying `torchvision` transforms. |
| **Scikit-learn** | Evaluation metrics (`accuracy_score`, `precision/recall/f1_score`, `classification_report`, `confusion_matrix`) and per-epoch macro F1 computation during training (`train.py`). Preferred over computing metrics manually: these are well-tested, standard implementations from the literature, including `zero_division` handling for classes with no predictions. |
| **Matplotlib** | Generates all charts saved in `results/` — training curves, confusion matrix, class distribution and the final comparison chart (`compare_results.py`). |

## Project structure

```text
IA2/
│
├── models/
│   ├── resnet.py
│   ├── googlenet.py
│   └── mobilenet.py
│
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   ├── inspect_dataset.py
│   ├── visualize_dataset.py
│   └── compare_results.py
│
├── results/
│   ├── resnet/
│   ├── googlenet/
│   └── mobilenet/
│
├── config.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `results/` folder **does not need to be created manually**. The code automatically creates the necessary directories at runtime.
