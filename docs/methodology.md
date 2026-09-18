# Técnicas de IA utilizadas — base teórica

Esta seção resume as técnicas empregadas no pipeline e **por que** cada uma foi escolhida neste experimento especificamente, servindo como rascunho para a seção de metodologia do artigo.

## Transfer Learning

* **Pesos pré-treinados na ImageNet** para os três backbones. Evita treinar do zero com um dataset de tamanho moderado (~156 mil imagens no total) e acelera a convergência, já que filtros de baixo/médio nível (bordas, texturas, formas) aprendidos na ImageNet são reaproveitáveis para imagens de galáxias.
* **Freeze de backbone configurável** (`FREEZE_BACKBONE`). Atualmente `False`: a rede inteira é ajustada (*fine-tuning* completo), permitindo que as próprias features convolucionais se especializem em morfologia de galáxia — mudança feita após o backbone congelado limitar o ResNet18 a ~41% de acurácia (ver [Changelog](changelog.md)). A opção de feature extraction (`True`, mais barata em tempo/memória, mas com teto de desempenho mais baixo) continua disponível para quem quiser comparar o trade-off.

## Balanceamento de classes

* **Pesos de classe na função de perda** (`nn.CrossEntropyLoss(weight=...)`), calculados a partir da frequência de cada classe no split de treino. Necessário porque classes como `Irregular` e `Merger` são bem mais raras no Galaxy Zoo Dataset que `Round Elliptical` — sem isso, o modelo tenderia a ignorar as classes minoritárias.

## Data Augmentation

* `RandomHorizontalFlip` e `RandomRotation(10)` — galáxias não têm uma orientação "correta"; a orientação da imagem é um artefato da captura, não uma característica da classe.
* `ColorJitter(brightness, contrast)` — simula variações de exposição entre observações astronômicas.
* Aplicado **só no split de treino**; validação e teste usam transformação sem augmentation, para medir desempenho em condições realistas.

## Normalização

* `Normalize` com média/desvio-padrão da ImageNet — obrigatório para transfer learning: a distribuição de entrada precisa bater com a que os pesos pré-treinados "esperam".

## Regularização

* **Weight decay** (AdamW) — penaliza pesos grandes, reduz overfitting.
* **Early stopping** (`EARLY_STOPPING_PATIENCE`) — interrompe o treino quando o F1 macro de validação para de melhorar, evitando treino desnecessário e overfitting tardio.
* **Batch Normalization** e **Dropout**, herdados das arquiteturas do Torchvision — não implementados por nós, mas ativos e relevantes para a estabilidade do treino.

## Loss auxiliar (GoogLeNet)

* Os classificadores auxiliares (`aux1`, `aux2`) contribuem com peso `0.3` cada na loss de treino. Técnica original da arquitetura Inception/GoogLeNet para injetar gradiente em camadas intermediárias e mitigar vanishing gradient em redes profundas. Usados só no treino; validação e teste usam exclusivamente a saída principal, para uma avaliação justa e comparável às outras arquiteturas.

## Otimização

* **AdamW** em vez de Adam — desacopla o weight decay do gradiente adaptativo, mais correto teoricamente que L2 embutido no Adam clássico.
* O otimizador recebe apenas parâmetros com `requires_grad=True`, coerente com o freeze de backbone — evita desperdiçar memória/computação com parâmetros congelados.

## Critério de seleção de modelo

* **F1 Macro de validação** (não accuracy) como critério para salvar o melhor checkpoint e para early stopping. Accuracy pode mascarar desempenho ruim em classes minoritárias; F1 macro pondera todas as classes igualmente, mais alinhado ao objetivo de comparar as arquiteturas de forma justa num dataset desbalanceado.

## Avaliação

* Métricas macro e weighted (precision, recall, F1), `classification_report` por classe e matriz de confusão — permitem diagnosticar *onde* cada modelo erra, não só *quanto*.
* Split em train/validation/test, com o conjunto de teste usado **exclusivamente** na avaliação final, nunca durante o treino ou tuning.

## Reprodutibilidade

* Seed fixa aplicada a todas as bibliotecas relevantes (`random`, `numpy`, `torch`, `torch.cuda`) e `cudnn` em modo determinístico — garante que os três modelos sejam comparados sob exatamente as mesmas condições experimentais.

---

# Reprodutibilidade

O projeto utiliza uma seed fixa:

```python
SEED = 42
```

A seed é aplicada às principais bibliotecas utilizadas no treinamento (`random`, `numpy`, `torch`, `torch.cuda`), além de `cudnn.deterministic = True` e `cudnn.benchmark = False`.

Além disso, os três modelos utilizam a mesma configuração experimental, permitindo uma comparação mais justa entre as arquiteturas.
