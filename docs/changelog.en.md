# Changes

## FREEZE_BACKBONE: True → False (full fine-tuning)

**Reason:** with the backbone frozen (feature extraction), ResNet18 reached 41.2% test accuracy (35.4% macro F1) — better than the trivial baseline of always predicting the majority class (~25%), but the training curves showed validation accuracy/F1 flattening out already around epoch 4-5, with validation macro F1 even slightly worsening afterward, while training kept slowly climbing. This is a sign of a **capacity ceiling**, not overfitting: a linear classifier on top of fixed ImageNet features (photos of everyday objects) struggles to capture the subtle differences that define galaxy morphology — e.g., the continuous spectrum between "Round", "In-between" and "Cigar-shaped" Elliptical, or the presence/absence of a central bar in spirals.

**What changes:** with `FREEZE_BACKBONE = False`, all convolutional backbone layers go back to having `requires_grad = True` (the code responsible for this is commented in `models/resnet.py`, `models/googlenet.py` and `models/mobilenet.py`), letting the features themselves specialize in galaxy textures and shapes during training, instead of staying locked to what ImageNet taught them.

**Actual result, measured on ResNet18 (before → after):**

| Metric | Frozen backbone | Full fine-tuning | Difference |
|---|---|---|---|
| Accuracy | 41.2% | 78.7% | **+37.5 p.p.** |
| F1 Macro | 35.4% | 72.4% | **+37.0 p.p.** |
| Training time | 2,042 s | 2,511 s | +23% |
| Peak GPU memory | 0.27 GB | 0.84 GB | +214% (still low in absolute terms) |
| Trainable parameters | 4,104 | 11,180,616 | 100% of the network |

The performance gain was much larger than the added cost — confirming the bottleneck was indeed the frozen backbone's lack of capacity, not hyperparameters or data quality. The rare classes (`Irregular`, `Merger`) had a large gain in *recall*, but still have low *precision* — an open point for a possible future iteration.

**Trade-offs to monitor when running GoogLeNet and MobileNet with the same configuration:**

* **Training time** and **GPU memory** should increase in a similar way (proportional to each architecture's size) — compare `training.training_time_seconds` and `training.max_gpu_memory_gb` in each model's `metrics.json`.
* **Overfitting risk**: in the ResNet curves, validation started to oscillate slightly around epoch 5-6 while training kept improving — early stopping cut it off in time, but it's worth watching the same pattern in the other two models; if the gap grows further, the first adjustments would be `WEIGHT_DECAY` or `EARLY_STOPPING_PATIENCE`.
* This change applies to all **three models** (the flag is global in `config.py`), so the final comparison between ResNet, GoogLeNet and MobileNet now reflects full fine-tuning, not feature extraction anymore.

## `training_summary.json` being overwritten between runs (fixed bug)

**Problem:** `main.py` allows training one model at a time (`--model resnet`, then `--model googlenet`, then `--model mobilenet`, in separate runs). However, the `save_comparison()` function only saved the list of models run **in that specific execution** to `training_summary.json` — so running `--model googlenet` after having already run `--model resnet` would overwrite the entire file with just the GoogLeNet result, erasing the ResNet one.

**Fix:** `save_comparison()` now reads the existing `training_summary.json` (if any), merges the results by model name — the new result replaces an old entry for the same model, but preserves results from different models already saved — and only then rewrites the file. In practice: running the three models across three separate `main.py` runs now accumulates all three in `training_summary.json`, instead of keeping only the last one.

**No manual action needed**, but if you already ran ResNet and then GoogLeNet/MobileNet *before* this fix, the current `training_summary.json` only has the result from the last run — each individual model's `metrics.json` (`results/<model>/metrics.json`), used by `compare_results.py`, is not affected by this bug and remains correct.

## `.gitignore`: `results/` now versioned, except the `.pth` checkpoints

**Reason:** `results/` was entirely in `.gitignore`, which prevented sharing each model's metrics, curves and confusion matrix with the rest of the team through the repository itself. At the same time, each model's checkpoints (`best_model.pth`) weigh ~40-60 MB each — versioning that weight in Git permanently bloats the repository's history, even if the files are later retrained/replaced.

**What changes:** `results/` is now versioned normally (metrics in `metrics.json`/`history.json`, `.png` charts), but `.pth` files in any subfolder of `results/` remain ignored:

```gitignore
results/**/*.pth
```

**How this affects day-to-day work:** `git add results/` now automatically includes metrics and charts, with no need for `git add -f` or listing files one by one — only `.pth` files stay out by default. If it's ever necessary to share a specific checkpoint (for inference without retraining), it can be added manually with `git add -f path/to/best_model.pth`, or versioned via Git LFS.
