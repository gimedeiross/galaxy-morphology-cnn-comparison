# Configuração

Os principais parâmetros do experimento estão centralizados em:

```text
config.py
```

A configuração utilizada nos experimentos é baseada em:

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

Os três modelos utilizam a mesma configuração experimental, permitindo uma comparação mais justa entre as arquiteturas.

---

# Estratégia de treinamento — Transfer Learning

Os três modelos utilizam **transfer learning** a partir de pesos pré-treinados na ImageNet.

Dois parâmetros em `config.py` controlam o comportamento:

```python
PRETRAINED = True

FREEZE_BACKBONE = False
```

* **`PRETRAINED = True`** — cada arquitetura é inicializada com os pesos pré-treinados na ImageNet (`ResNet18_Weights.DEFAULT`, `GoogLeNet_Weights.DEFAULT`, `MobileNet_V3_Small_Weights.DEFAULT`), em vez de pesos aleatórios.
* **`FREEZE_BACKBONE = False`** (configuração atual) — o backbone continua inicializado com pesos da ImageNet, mas toda a rede é treinada (**fine-tuning**), ajustando também as camadas convolucionais pré-treinadas ao domínio do Galaxy Zoo. Ver o [Changelog](changelog.md) para o motivo da mudança.

Se `FREEZE_BACKBONE = True`, o extrator de características (backbone) é congelado (`requires_grad = False`), e apenas a cabeça de classificação — recriada para as 8 classes do Galaxy Zoo — é treinada. Essa técnica é conhecida como **feature extraction**, e foi a configuração original deste projeto.

O otimizador (`AdamW`) recebe apenas os parâmetros com `requires_grad = True`. Com `FREEZE_BACKBONE = False`, isso passa a incluir praticamente todos os parâmetros do modelo (não só a cabeça), aumentando o custo computacional do treinamento — ver o [Changelog](changelog.md).

A quantidade de parâmetros treináveis (em relação ao total) é impressa no console a cada modelo treinado, e também fica registrada em `metrics.json` (`parameters.total`), permitindo comparar o "tamanho efetivo" do treinamento entre os três modelos.
