# Dataset

The project uses:

```text
mrJordi0/galaxy-zoo-dataset
```

The dataset is loaded automatically through the Hugging Face Datasets library:

```python
load_dataset(DATASET_NAME)
```

There's no need to manually download the files or add them to the repository.

On the first run, the dataset files will be downloaded to the local Hugging Face cache. Subsequent runs will use the cached files.

## Data split

The dataset already comes with three splits:

```text
Train:      99,808 images
Validation: 24,952 images
Test:       31,191 images
```

The code uses:

* `train` for training;
* `validation` for monitoring during training and early stopping;
* `test` exclusively for the final evaluation.

---

# Classes

The dataset has eight classes:

| ID | Class                   |
| -: | ----------------------- |
|  0 | Round Elliptical        |
|  1 | In-between Elliptical   |
|  2 | Cigar-shaped Elliptical |
|  3 | Edge-on Spiral          |
|  4 | Barred Spiral           |
|  5 | Unbarred Spiral         |
|  6 | Irregular               |
|  7 | Merger                  |

There is class imbalance, especially for the `Irregular` and `Merger` classes.

Because of this, training uses **class weights** in the loss function to reduce the impact of the imbalance.

---

# Inspecting the dataset

Before starting the experiments, it's recommended to check the dataset's structure and distribution:

```bash
python -m src.inspect_dataset
```

The `inspect_dataset.py` script performs a basic inspection of the dataset and reports information such as:

* number of examples per split;
* available features;
* image size and format;
* class distribution;
* percentage of each class.

This script is mainly used to **validate the dataset before training**, avoiding long experiments running on incorrectly loaded or structured data.

---

# Visualizing the dataset

It's also possible to generate some visualizations of the images before training:

```bash
python -m src.visualize_dataset
```

The `visualize_dataset.py` script produces visualizations to help with initial data analysis.

It produces:

* sample images from each of the eight classes;
* a chart showing the class distribution in the training set.

Files are saved to:

```text
results/
├── dataset_samples.png
└── class_distribution.png
```

These visualizations make it possible to visually check the classes and identify the dataset's imbalance before running the models.
