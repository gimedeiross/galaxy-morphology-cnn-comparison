# Dataset

O projeto utiliza:

```text
mrJordi0/galaxy-zoo-dataset
```

O dataset é carregado automaticamente através da biblioteca Hugging Face Datasets:

```python
load_dataset(DATASET_NAME)
```

Não é necessário baixar manualmente os arquivos nem adicioná-los ao repositório.

Na primeira execução, os arquivos do dataset serão baixados para o cache local do Hugging Face. Execuções posteriores utilizarão os arquivos armazenados em cache.

## Divisão dos dados

O dataset utilizado já possui três divisões:

```text
Train:      99.808 imagens
Validation: 24.952 imagens
Test:       31.191 imagens
```

O código utiliza:

* `train` para treinamento;
* `validation` para acompanhamento durante o treinamento e early stopping;
* `test` exclusivamente para avaliação final.

---

# Classes

O dataset possui oito classes:

| ID | Classe                  |
| -: | ----------------------- |
|  0 | Round Elliptical        |
|  1 | In-between Elliptical   |
|  2 | Cigar-shaped Elliptical |
|  3 | Edge-on Spiral          |
|  4 | Barred Spiral           |
|  5 | Unbarred Spiral         |
|  6 | Irregular               |
|  7 | Merger                  |

Existe um desbalanceamento entre as classes, especialmente nas classes `Irregular` e `Merger`.

Por isso, o treinamento utiliza **pesos de classe** na função de perda para reduzir o impacto do desbalanceamento.

---

# Inspecionando o dataset

Antes de iniciar os experimentos, é recomendado verificar a estrutura e a distribuição do dataset:

```bash
python -m src.inspect_dataset
```

O script `inspect_dataset.py` realiza uma inspeção básica do dataset e apresenta informações como:

* quantidade de exemplos em cada split;
* features disponíveis;
* tamanho e formato das imagens;
* distribuição das classes;
* percentual de cada classe.

Esse script serve principalmente para **validar o dataset antes do treinamento**, evitando iniciar experimentos longos com dados carregados ou estruturados incorretamente.

---

# Visualizando o dataset

Também é possível gerar algumas visualizações das imagens antes do treinamento:

```bash
python -m src.visualize_dataset
```

O script `visualize_dataset.py` gera visualizações para facilitar a análise inicial dos dados.

São produzidos:

* exemplos de imagens de cada uma das oito classes;
* gráfico com a distribuição das classes no conjunto de treinamento.

Os arquivos são salvos em:

```text
results/
├── dataset_samples.png
└── class_distribution.png
```

Essas visualizações permitem verificar visualmente as classes e identificar o desbalanceamento do dataset antes da execução dos modelos.
