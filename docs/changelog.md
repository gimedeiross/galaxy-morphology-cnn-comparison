# Alterações

## FREEZE_BACKBONE: True → False (fine-tuning completo)

**Motivo:** com o backbone congelado (feature extraction), o ResNet18 atingiu 41,2% de acurácia no teste (F1 macro 35,4%) — melhor que o baseline trivial de sempre prever a classe majoritária (~25%), mas as curvas de treino mostravam a acurácia/F1 de validação achatando já por volta do epoch 4-5, com o F1 macro de validação até piorando levemente depois disso, enquanto o treino seguia subindo devagar. Isso é sinal de **teto de capacidade**, não de overfitting: um classificador linear sobre features fixas da ImageNet (fotos de objetos do dia a dia) tem dificuldade em capturar as diferenças sutis que definem morfologia de galáxia — ex.: o espectro contínuo entre "Round", "In-between" e "Cigar-shaped" Elliptical, ou a presença/ausência de barra central nas espirais.

**O que muda:** com `FREEZE_BACKBONE = False`, todas as camadas convolucionais do backbone voltam a ter `requires_grad = True` (a lógica de qual código faz isso está comentada em `models/resnet.py`, `models/googlenet.py` e `models/mobilenet.py`), permitindo que as próprias features se especializem em texturas e formas de galáxias durante o treino, em vez de ficarem presas ao que a ImageNet ensinou.

**Resultado real, medido no ResNet18 (antes → depois):**

| Métrica | Backbone congelado | Fine-tuning completo | Diferença |
|---|---|---|---|
| Accuracy | 41,2% | 78,7% | **+37,5 p.p.** |
| F1 Macro | 35,4% | 72,4% | **+37,0 p.p.** |
| Tempo de treino | 2.042 s | 2.511 s | +23% |
| Memória GPU máxima | 0,27 GB | 0,84 GB | +214% (ainda baixo em absoluto) |
| Parâmetros treináveis | 4.104 | 11.180.616 | 100% da rede |

O ganho de desempenho foi bem maior do que o custo adicional — confirma que o gargalo era mesmo falta de capacidade do backbone congelado, não hiperparâmetros ou qualidade do dado. As classes raras (`Irregular`, `Merger`) tiveram grande ganho de *recall*, mas ainda têm *precision* baixa — ponto em aberto para uma eventual próxima iteração.

**Trade-offs a monitorar ao rodar GoogLeNet e MobileNet com essa mesma configuração:**

* **Tempo de treino** e **memória GPU** devem aumentar de forma parecida (proporcionalmente ao tamanho de cada arquitetura) — comparar `training.training_time_seconds` e `training.max_gpu_memory_gb` no `metrics.json` de cada modelo.
* **Risco de overfitting**: nas curvas do ResNet, a validação começou a oscilar levemente por volta do epoch 5-6 enquanto o treino seguia melhorando — o early stopping cortou a tempo, mas vale acompanhar o mesmo padrão nos outros dois modelos; se o gap crescer mais, os primeiros ajustes seriam `WEIGHT_DECAY` ou `EARLY_STOPPING_PATIENCE`.
* Essa alteração vale para os **três modelos** (a flag é global em `config.py`), então a comparação final entre ResNet, GoogLeNet e MobileNet passa a refletir fine-tuning completo, não mais feature extraction.

## `training_summary.json` sendo sobrescrito entre execuções (bug corrigido)

**Problema:** `main.py` permite rodar um modelo por vez (`--model resnet`, depois `--model googlenet`, depois `--model mobilenet`, em execuções separadas). A função `save_comparison()`, porém, salvava em `training_summary.json` apenas a lista de modelos rodados **naquela execução específica** — então rodar `--model googlenet` depois de já ter rodado `--model resnet` sobrescrevia o arquivo inteiro só com o resultado do GoogLeNet, apagando o do ResNet.

**Correção:** `save_comparison()` agora lê o `training_summary.json` existente (se houver), mescla os resultados por nome de modelo — o resultado novo substitui uma entrada antiga do mesmo modelo, mas preserva os resultados de modelos diferentes já salvos — e só então regrava o arquivo. Na prática: rodar os três modelos em três execuções separadas do `main.py` agora acumula os três no `training_summary.json`, em vez de manter só o último.

**Não precisa fazer nada manualmente**, mas se você já rodou ResNet e depois GoogLeNet/MobileNet *antes* dessa correção, o `training_summary.json` atual só tem o resultado da última execução — o `metrics.json` de cada modelo individual (`results/<modelo>/metrics.json`), usado pelo `compare_results.py`, não é afetado por esse bug e continua correto.

## `.gitignore`: `results/` versionado, exceto os checkpoints `.pth`

**Motivo:** `results/` estava inteiramente no `.gitignore`, o que impedia compartilhar métricas, curvas e a matriz de confusão de cada modelo com o resto da equipe pelo próprio repositório. Ao mesmo tempo, os checkpoints (`best_model.pth`) de cada modelo pesam ~40-60 MB cada — versionar esse peso no Git infla o histórico do repositório permanentemente, mesmo que os arquivos sejam retreinados/substituídos depois.

**O que muda:** `results/` passa a ser versionado normalmente (métricas em `metrics.json`/`history.json`, gráficos `.png`), mas os arquivos `.pth` de qualquer subpasta de `results/` continuam ignorados:

```gitignore
results/**/*.pth
```

**Como isso afeta o dia a dia:** `git add results/` agora inclui métricas e gráficos automaticamente, sem precisar de `git add -f` nem listar arquivo por arquivo — só o `.pth` fica de fora, por padrão. Se algum dia for necessário compartilhar um checkpoint específico (para inferência sem re-treinar), ele pode ser adicionado manualmente com `git add -f caminho/para/best_model.pth`, ou versionado via Git LFS.
