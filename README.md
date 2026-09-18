# Galaxy Zoo — CNN Architecture Comparison

[🇧🇷 Leia em português](README.pt-BR.md)

Project for the **Artificial Intelligence II** course: training, evaluating, and comparing three CNN architectures (**ResNet18**, **GoogLeNet**, and **MobileNetV3 Small**) for galaxy classification, using transfer learning from weights pre-trained on ImageNet.

📖 **Complete documentation:** [GitHub Pages link](https://gimedeiross.github.io/galaxy-morphology-cnn-comparison/en/)

## About

The project uses the [`mrJordi0/galaxy-zoo-dataset`](https://huggingface.co/datasets/mrJordi0/galaxy-zoo-dataset) dataset (Hugging Face) to classify galaxies into 8 morphological classes, comparing the three architectures under the same experimental conditions (fixed seed, same split, same training configuration), collecting performance and computational cost metrics for use in the scientific paper.

## Quick Installation

```bash
git clone https://github.com/gimedeiross/galaxy-morphology-cnn-comparison

cd galaxy-morphology-cnn-comparison/

python3 -m venv .venv

source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

```bash
# (optional) inspect and visualize the dataset

python -m src.inspect_dataset

python -m src.visualize_dataset

# train all models (default)

python main.py --model all

# or train a specific model

python main.py --model resnet   # | googlenet | mobilenet

# compare the results (after training all three models)

python -m src.compare_results
```

Results (metrics, checkpoints, and plots) are automatically saved in `results/<model>/`.

## Structure

```text
galaxy-morphology-cnn-comparison//
├── models/       # definition of the 3 architectures
├── src/          # dataset, training, evaluation, comparison
├── results/      # generated metrics and plots (created automatically)
├── config.py     # experiment hyperparameters and flags
└── main.py       # entry point
```

## Documentation

For details about the dataset, configuration, each architecture, collected metrics, the theoretical basis of the techniques used, and the project's change history, see the complete documentation: [GitHub Pages link](https://gimedeiross.github.io/galaxy-morphology-cnn-comparison/en/)

## License

Academic project developed for the Artificial Intelligence II course.
