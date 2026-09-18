# Training

`main.py` accepts a `--model` flag to choose what to train:

## Train a single model (useful to quickly test whether the pipeline works)

```bash
python main.py --model resnet
```

```bash
python main.py --model googlenet
```

```bash
python main.py --model mobilenet
```

## Train all three models sequentially

```bash
python main.py --model all
```

`--model all` is the **default** value — running `python main.py` with no flags also trains all three models, in the order defined in `config.MODELS`:

```text
ResNet18
    ↓
GoogLeNet
    ↓
MobileNetV3 Small
```

Each model has its own results directory (`results/<model>/`), regardless of whether it was trained alone or alongside the others.

---

# What happens during a run?

When running (for example):

```bash
python main.py --model all
```

the program:

1. loads the dataset;
2. prepares the DataLoaders;
3. creates each selected architecture, loading ImageNet-pretrained weights;
4. freezes the backbone (if `FREEZE_BACKBONE = True`), keeping only the classification head trainable;
5. sets up the loss function with class weights;
6. trains the model, computing accuracy and macro F1 per epoch on train and validation;
7. evaluates on the validation set every epoch and applies early stopping when necessary;
8. saves a checkpoint whenever the validation macro F1 improves;
9. reloads the best saved checkpoint at the end of training;
10. evaluates the best model on the test set;
11. computes the final metrics;
12. generates charts (loss, accuracy, macro F1 and confusion matrix);
13. saves the results as JSON;
14. once all selected models are done, saves a consolidated summary (`training_summary.json`).

---

# Experiment flow

```text
                  Galaxy Zoo Dataset
                         │
                         ▼
                 Dataset inspection
                         │
                         ▼
                Dataset visualization
                         │
                         ▼
                   Preprocessing
                         │
                         ▼
                     DataLoader
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          ResNet18    GoogLeNet   MobileNetV3
             │           │           │
             └───────────┼───────────┘
                         ▼
        Pretrained weights (ImageNet)
                         │
                         ▼
          Backbone freeze (optional)
                         │
                         ▼
                     Training
                         │
                         ▼
                  Best checkpoint
                    (F1 Macro)
                         │
                         ▼
                   Final test
                         │
                         ▼
                      Metrics
                         │
                         ▼
                Matrices and charts
                         │
                         ▼
                 Final comparison
                         │
                         ▼
                 Scientific paper
```

---

# Essential commands

### Check the dataset

```bash
python -m src.inspect_dataset
```

### Visualize the dataset

```bash
python -m src.visualize_dataset
```

### Train a single model (quick pipeline test)

```bash
python main.py --model resnet
```

```bash
python main.py --model googlenet
```

```bash
python main.py --model mobilenet
```

### Train all (default)

```bash
python main.py
```

or, explicitly:

```bash
python main.py --model all
```

### Compare results (after training all three models)

```bash
python -m src.compare_results
```
