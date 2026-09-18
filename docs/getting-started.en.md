# Installation

## 1. Clone the repository

```bash
git clone https://github.com/gimedeiross/galaxy-morphology-cnn-comparison

cd galaxy-morphology-cnn-comparison/
```

## 2. Create the virtual environment

```bash
python3 -m venv .venv
```

## 3. Activate the virtual environment

### Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 4. Install the dependencies

```bash
pip install -r requirements.txt
```

---

# NVIDIA GPU

Before starting the experiments, it's recommended to check whether PyTorch is detecting the GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

On a machine with CUDA properly configured, the output should show:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

If `torch.cuda.is_available()` returns `False`, training will run on the CPU.
