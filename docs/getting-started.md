# Instalação

## 1. Clonar o repositório

```bash
git clone https://github.com/gimedeiross/galaxy-morphology-cnn-comparison

cd galaxy-morphology-cnn-comparison/
```

## 2. Criar o ambiente virtual

```bash
python3 -m venv .venv
```

## 3. Ativar o ambiente virtual

### Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 4. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

# GPU NVIDIA

Antes de iniciar os experimentos, é recomendado verificar se o PyTorch está reconhecendo a GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponível:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Em uma máquina com CUDA configurada corretamente, o resultado deverá indicar:

```text
CUDA disponível: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

Caso `torch.cuda.is_available()` retorne `False`, o treinamento será executado na CPU.
