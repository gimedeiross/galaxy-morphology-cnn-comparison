# Modelos

## ResNet18

Implementada utilizando a arquitetura disponibilizada pelo Torchvision:

```text
ResNet18
```

Inicializada com pesos pré-treinados na ImageNet (`ResNet18_Weights.DEFAULT`).

A camada final (`fc`) é substituída para produzir oito classes; essa nova camada é sempre treinável, independentemente de `FREEZE_BACKBONE`.

---

## GoogLeNet

Implementada utilizando:

```text
GoogLeNet
```

Inicializada com pesos pré-treinados na ImageNet (`GoogLeNet_Weights.DEFAULT`).

O modelo possui dois classificadores auxiliares (`aux1` e `aux2`), além do classificador principal.

Todos os classificadores (`fc`, `aux1.fc2`, `aux2.fc2`) são adaptados para produzir oito classes e permanecem sempre treináveis, mesmo com o restante do backbone congelado.

Durante o **treinamento**, a função de perda considera:

```text
Loss =
    Loss principal
    + 0.3 × Loss auxiliar 1
    + 0.3 × Loss auxiliar 2
```

Durante a **validação e a avaliação final**, somente a saída principal é utilizada.

---

## MobileNetV3 Small

Implementada utilizando:

```text
MobileNetV3 Small
```

Inicializada com pesos pré-treinados na ImageNet (`MobileNet_V3_Small_Weights.DEFAULT`).

Quando o backbone é congelado, apenas `model.features` (o extrator convolucional) tem os pesos congelados — todo o `model.classifier` (não só a última camada) permanece treinável, já que a cabeça do MobileNetV3 é composta por múltiplas camadas (`Linear → Hardswish → Dropout → Linear`) que costumam se beneficiar de treinar juntas.

A MobileNetV3 Small foi escolhida por possuir uma arquitetura consideravelmente mais leve, permitindo comparar não apenas o desempenho de classificação, mas também o custo computacional e a quantidade de parâmetros.

---

# Funções de ativação por modelo

O que muda entre as três arquiteturas em termos de funções de ativação — no backbone (herdado do torchvision, pré-treinado) e na cabeça de classificação (código nosso, em `models/*.py`).

| Modelo | Ativação no backbone | Ativação na cabeça nova |
|---|---|---|
| **ResNet18** | ReLU (inplace), após cada BatchNorm dentro dos blocos residuais. Padrão da arquitetura original, não alterado. | Nenhuma — `model.fc` é uma única `nn.Linear`. Classificador linear puro sobre as features. |
| **GoogLeNet** | ReLU dentro de cada módulo Inception, inclusive nos branches auxiliares (`aux1`, `aux2`). Padrão da arquitetura original, não alterado. | Nenhuma — `fc`, `aux1.fc2` e `aux2.fc2` são `nn.Linear` puras. |
| **MobileNetV3-Small** | Mista: ReLU nas camadas iniciais de `features` (mais barata) e Hardswish nas camadas finais (mais expressiva). Decisão de design do paper original, não alterada. | Hardswish — herdada do `classifier` original (`Linear -> Hardswish -> Dropout -> Linear`); só a última `Linear` é substituída, então a Hardswish intermediária permanece ativa. |

**Pontos a destacar:**

* Nenhuma ativação é adicionada ou escolhida por nós — todas vêm de dentro dos backbones pré-treinados do torchvision. O único código que decide algo sobre ativação é a escolha de **não** adicionar nenhuma às cabeças novas do ResNet e do GoogLeNet, versus **herdar** a Hardswish já existente na cabeça do MobileNet.
* Isso cria uma assimetria entre os três modelos: o MobileNet tem uma cabeça não-linear "de fábrica", enquanto ResNet e GoogLeNet ficam com *linear probing* puro (comum em transfer learning com backbone congelado). Vale mencionar essa diferença na metodologia do artigo, já que pode influenciar a comparação — não é só o backbone que difere entre os modelos, a capacidade da cabeça também difere.
* Os comentários equivalentes a esta tabela estão nos docstrings de `create_model()` em `models/resnet.py`, `models/googlenet.py` e `models/mobilenet.py`.
