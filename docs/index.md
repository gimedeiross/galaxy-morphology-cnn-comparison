# Galaxy Zoo — Comparação de Arquiteturas CNN

Projeto da disciplina de **Inteligência Artificial II** para treinamento, avaliação e comparação de arquiteturas de Redes Neurais Convolucionais aplicadas à classificação de galáxias.

O projeto utiliza o dataset **`mrJordi0/galaxy-zoo-dataset`**, disponibilizado pelo Hugging Face, e compara três arquiteturas:

* **ResNet18**
* **GoogLeNet**
* **MobileNetV3 Small**

Os três modelos utilizam **transfer learning**: partem de pesos pré-treinados na ImageNet e, atualmente, treinam a rede inteira (fine-tuning completo — ver [Changelog](changelog.md)). O projeto também suporta a variante mais barata de treinar só a cabeça de classificação (backbone congelado), configurável em `config.py`.

O objetivo é realizar os experimentos sob condições controladas e coletar métricas que possam ser utilizadas na elaboração do artigo científico.

## Objetivo do projeto

O objetivo não é apenas identificar qual arquitetura apresenta a maior acurácia. A análise pretende comparar as arquiteturas considerando:

* desempenho de classificação;
* F1 Macro;
* F1 Weighted;
* desempenho por classe;
* matriz de confusão;
* comportamento durante o treinamento;
* quantidade de parâmetros totais e treináveis (relevante com backbone congelado);
* tempo de treinamento;
* tempo de avaliação;
* utilização de memória da GPU.

Dessa forma, os resultados poderão ser utilizados para discutir os **trade-offs entre desempenho e custo computacional** das arquiteturas avaliadas no artigo científico de Inteligência Artificial II.

## Tecnologias

| Biblioteca | Por que é usada neste projeto |
|---|---|
| **PyTorch** | Framework de deep learning usado para os três modelos, o treino, a loss e o otimizador. Escolha padrão para trabalhar com `torchvision` e ter controle explícito do loop de treino (útil aqui porque GoogLeNet precisa de um tratamento especial para as saídas auxiliares — ver `train.py`). |
| **Torchvision** | Fornece as três arquiteturas (`resnet18`, `googlenet`, `mobilenet_v3_small`) já com pesos pré-treinados na ImageNet (`*_Weights.DEFAULT`), evitando reimplementar as redes e permitindo transfer learning direto. |
| **Hugging Face Datasets** | Carrega o dataset `mrJordi0/galaxy-zoo-dataset` diretamente do Hub (`load_dataset`), com cache local automático e suporte a `train_test_split` para gerar o split de validação. Preferido a baixar/organizar os arquivos manualmente em pastas (como pediria `torchvision.datasets.ImageFolder`), já que o dataset já é distribuído nesse formato. |
| **NumPy** | Suporte numérico usado indiretamente por `torch`, `sklearn` e `matplotlib` (ex.: `np.arange` nos ticks da matriz de confusão em `evaluate.py`). |
| **Pillow** | Manipulação das imagens carregadas pelo `datasets` (`.convert("RGB")` em `dataset.py`) antes de aplicar as transformações do `torchvision`. |
| **Scikit-learn** | Métricas de avaliação (`accuracy_score`, `precision/recall/f1_score`, `classification_report`, `confusion_matrix`) e cálculo do F1 macro por época durante o treino (`train.py`). Preferido a calcular as métricas manualmente: são implementações testadas e padrão da literatura, incluindo o tratamento de `zero_division` para classes sem previsões. |
| **Matplotlib** | Geração de todos os gráficos salvos em `results/` — curvas de treino, matriz de confusão, distribuição de classes e o gráfico comparativo final (`compare_results.py`). |

## Estrutura do projeto

```text
galaxy-morphology-cnn-comparison/
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

A pasta `results/` **não precisa ser criada manualmente**. O código cria os diretórios necessários automaticamente durante a execução.
