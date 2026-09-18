# Métricas coletadas

O projeto coleta métricas durante o treinamento e durante a avaliação final.

## Durante o treinamento

São registradas por época:

* Training Loss / Validation Loss;
* Training Accuracy / Validation Accuracy;
* Training F1 Macro / Validation F1 Macro;
* tempo por época;
* tempo total de treinamento;
* melhor Validation F1 Macro e a Validation Accuracy correspondente;
* número de épocas executadas;
* memória máxima utilizada pela GPU (`max_gpu_memory_gb`).

O **F1 Macro de validação** — e não a accuracy — é o critério usado para decidir qual checkpoint salvar e para o early stopping, por ser mais robusto ao desbalanceamento entre classes.

## Avaliação final

São calculadas, no conjunto de teste:

* Accuracy;
* Precision Weighted;
* Recall Weighted;
* F1-score Weighted;
* Precision Macro;
* Recall Macro;
* F1-score Macro;
* Classification Report;
* Matriz de confusão.

O **F1 Macro** é especialmente importante neste projeto devido ao desbalanceamento das classes, pois atribui o mesmo peso a cada classe.

---

# Métricas de custo computacional

Também são registrados:

* número total de parâmetros;
* número de parâmetros treináveis (relevante especialmente com `FREEZE_BACKBONE = True`, quando esse número é bem menor que o total);
* tempo total de treinamento;
* tempo de avaliação;
* dispositivo utilizado;
* GPU utilizada;
* memória máxima utilizada pela GPU.

Essas informações permitem comparar não apenas qual modelo possui melhor desempenho, mas também qual apresenta melhor relação entre **desempenho e custo computacional**.

---

# Resultados

Os resultados são automaticamente armazenados em:

```text
results/
```

Para cada arquitetura:

```text
results/
├── resnet/
│   ├── best_model.pth
│   ├── metrics.json
│   ├── history.json
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── f1_macro_curve.png
│
├── googlenet/
│   ├── best_model.pth
│   ├── metrics.json
│   ├── history.json
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── f1_macro_curve.png
│
└── mobilenet/
    ├── best_model.pth
    ├── metrics.json
    ├── history.json
    ├── confusion_matrix.png
    ├── loss_curve.png
    ├── accuracy_curve.png
    └── f1_macro_curve.png
```

Ao final de uma execução do `main.py`, também é criado:

```text
results/training_summary.json
```

Esse arquivo consolida os resultados brutos dos modelos treinados **naquela execução** (um único modelo, se `--model resnet/googlenet/mobilenet` foi usado, ou os três, se `--model all`).

!!! warning "Atenção"
    `results/comparison.json` — a tabela comparativa formatada, usada para o artigo — **não** é gerado automaticamente pelo `main.py`. Ele é produzido separadamente por `compare_results.py` (veja a seção seguinte), que também exige que os três modelos já tenham sido treinados e seus `metrics.json` estejam presentes.

---

# Comparação dos resultados

Depois que os três modelos forem treinados (`python main.py --model all`, ou os três `--model <modelo>` individualmente), execute:

```bash
python -m src.compare_results
```

Esse script lê `results/<modelo>/metrics.json` de cada arquitetura, monta a tabela comparativa e imprime no console:

| Modelo      | Accuracy | F1 Macro | F1 Weighted | Parâmetros | Tempo |
| ----------- | -------: | -------: | ----------: | ---------: | ----: |
| ResNet18    |        - |        - |           - |          - |     - |
| GoogLeNet   |        - |        - |           - |          - |     - |
| MobileNetV3 |        - |        - |           - |          - |     - |

Os valores serão preenchidos após a execução dos experimentos.

Além da tabela, o script gera e salva:

```text
results/
├── comparison.json
└── model_comparison.png
```

---

# Arquivos importantes para o artigo

## `metrics.json`

Contém as métricas finais do modelo, organizadas em três blocos:

```json
{
  "model": "resnet",
  "metrics": { "accuracy": "...", "f1_macro": "...", "..." : "..." },
  "parameters": { "total": "...", "trainable": "..." },
  "training": {
    "training_time_seconds": "...",
    "evaluation_time_seconds": "...",
    "epochs_completed": "...",
    "best_validation_accuracy": "...",
    "best_validation_f1_macro": "...",
    "max_gpu_memory_gb": "...",
    "pretrained": true,
    "freeze_backbone": true,
    "class_weights": true
  }
}
```

Esse schema é o mesmo lido por `compare_results.py` para montar a tabela comparativa.

## `history.json`

Contém os dados obtidos durante cada época do treinamento (loss, accuracy e F1 macro de treino e validação).

Pode ser utilizado para analisar:

* convergência;
* overfitting;
* estabilidade do treinamento;
* evolução da loss, da accuracy e do F1 macro.

## `confusion_matrix.png`

Permite analisar quais classes são mais confundidas pelo modelo.

## `loss_curve.png` / `accuracy_curve.png` / `f1_macro_curve.png`

Mostram a evolução, por época, da loss, da accuracy e do F1 macro de treino e validação, respectivamente.

## `training_summary.json`

Gerado pelo `main.py` ao final de cada execução. Consolida em uma lista os resultados brutos (mesmo schema de `metrics.json`) de todos os modelos treinados naquela execução específica.

## `comparison.json`

Gerado pelo `compare_results.py` (execução separada, depois de treinar os três modelos). Consolida os resultados dos três modelos em uma tabela achatada e facilita a criação das tabelas comparativas do artigo.
