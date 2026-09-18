# Galaxy Zoo — Comparação de Arquiteturas CNN

[🇺🇸 Read in English](README.md)

Projeto da disciplina de **Inteligência Artificial II**: treinamento, avaliação e comparação de três arquiteturas de CNN (**ResNet18**, **GoogLeNet**, **MobileNetV3 Small**) na classificação de galáxias, usando transfer learning a partir de pesos pré-treinados na ImageNet.

📖 **Documentação completa:** [link do GitHub Pages](https://gimedeiross.github.io/galaxy-morphology-cnn-comparison/)

## Sobre

O projeto usa o dataset [`mrJordi0/galaxy-zoo-dataset`](https://huggingface.co/datasets/mrJordi0/galaxy-zoo-dataset) (Hugging Face) para classificar galáxias em 8 classes morfológicas, comparando as três arquiteturas sob as mesmas condições experimentais (seed fixa, mesmo split, mesma configuração de treino), coletando métricas de desempenho e de custo computacional para uso no artigo científico.

## Instalação rápida

```bash
git clone https://github.com/gimedeiross/galaxy-morphology-cnn-comparison
cd galaxy-morphology-cnn-comparison/

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Uso

```bash
# (opcional) inspecionar e visualizar o dataset
python -m src.inspect_dataset
python -m src.visualize_dataset

# treinar todos os modelos (padrão)
python main.py --model all

# ou treinar um modelo específico
python main.py --model resnet   # | googlenet | mobilenet

# comparar os resultados (após treinar os três)
python -m src.compare_results
```

Os resultados (métricas, checkpoints e gráficos) são salvos automaticamente em `results/<modelo>/`.

## Estrutura

```text
galaxy-morphology-cnn-comparison//
├── models/       # definição das 3 arquiteturas
├── src/          # dataset, treino, avaliação, comparação
├── results/      # métricas e gráficos gerados (criado automaticamente)
├── config.py     # hiperparâmetros e flags do experimento
└── main.py       # ponto de entrada
```

## Documentação

Para detalhes sobre o dataset, configuração, cada arquitetura, métricas coletadas, base teórica das técnicas usadas e o histórico de alterações do projeto, veja a documentação completa: [link do GitHub Pages](https://gimedeiross.github.io/galaxy-morphology-cnn-comparison/)

## Licença

Projeto acadêmico desenvolvido para a disciplina de Inteligência Artificial II.
