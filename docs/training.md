# Treinamento

O `main.py` aceita a flag `--model` para escolher o que treinar:

## Treinar um único modelo (útil para testar rapidamente se o pipeline está funcionando)

```bash
python main.py --model resnet
```

```bash
python main.py --model googlenet
```

```bash
python main.py --model mobilenet
```

## Treinar os três modelos sequencialmente

```bash
python main.py --model all
```

`--model all` é o valor **padrão** — rodar `python main.py` sem nenhuma flag também treina os três modelos, na ordem definida em `config.MODELS`:

```text
ResNet18
    ↓
GoogLeNet
    ↓
MobileNetV3 Small
```

Cada modelo possui seu próprio diretório de resultados (`results/<modelo>/`), independentemente de ter sido treinado sozinho ou em conjunto com os outros.

---

# O que acontece durante a execução?

Ao executar (por exemplo):

```bash
python main.py --model all
```

o programa:

1. carrega o dataset;
2. prepara os DataLoaders;
3. cria cada arquitetura selecionada, carregando pesos pré-treinados na ImageNet;
4. congela o backbone (se `FREEZE_BACKBONE = True`), mantendo treinável apenas a cabeça de classificação;
5. configura a função de perda com pesos de classe;
6. treina o modelo, calculando accuracy e F1 macro por época em treino e validação;
7. avalia no conjunto de validação a cada época e aplica early stopping quando necessário;
8. salva o checkpoint sempre que o F1 macro de validação melhora;
9. ao final do treino, recarrega o melhor checkpoint salvo;
10. avalia o melhor modelo no conjunto de teste;
11. calcula as métricas finais;
12. gera gráficos (loss, accuracy, F1 macro e matriz de confusão);
13. salva os resultados em JSON;
14. ao final de todos os modelos selecionados, salva um resumo consolidado (`training_summary.json`).

---

# Fluxo do experimento

```text
                  Galaxy Zoo Dataset
                         │
                         ▼
                  Inspeção dos dados
                         │
                         ▼
                  Visualização dos dados
                         │
                         ▼
                  Pré-processamento
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
        Pesos pré-treinados (ImageNet)
                         │
                         ▼
          Freeze do backbone (opcional)
                         │
                         ▼
                    Treinamento
                         │
                         ▼
                  Melhor checkpoint
                    (F1 Macro)
                         │
                         ▼
                    Teste final
                         │
                         ▼
                      Métricas
                         │
                         ▼
                Matrizes e gráficos
                         │
                         ▼
                  Comparação final
                         │
                         ▼
                  Artigo científico
```

---

# Comandos essenciais

### Verificar dataset

```bash
python -m src.inspect_dataset
```

### Visualizar dataset

```bash
python -m src.visualize_dataset
```

### Treinar apenas um modelo (teste rápido do pipeline)

```bash
python main.py --model resnet
```

```bash
python main.py --model googlenet
```

```bash
python main.py --model mobilenet
```

### Treinar todos (padrão)

```bash
python main.py
```

ou, de forma explícita:

```bash
python main.py --model all
```

### Comparar resultados (depois de treinar os três modelos)

```bash
python -m src.compare_results
```
