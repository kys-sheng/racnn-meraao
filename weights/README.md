# RACNN: trained weights

Trained models for **"Multitask, Every Region, All at Once: Region-Conditioned Simulation-Based Inference with Signal Detection and Region Evaluation at Colliders"** (Y. S. Koay, T. Glüsenkamp, S. Moretti, P. Singh).

The model code is in `bumphunt/` and `monojet/` of this repository; the datasets are on Hugging Face: [huggingface.co/datasets/kys-sheng/racnn-meraao](https://huggingface.co/datasets/kys-sheng/racnn-meraao).

## Files

```
LICENSE                                  CC BY 4.0
model_configs.json                       constructor arguments per analysis and model family
ensembles.json                           the 3 runs (of 5) forming each paper ensemble, and checkpoint meanings
bumphunt/<family>_<run>/<checkpoint>.safetensors
monojet/<family>_<run>/<checkpoint>.safetensors
```

**Families** (6 per analysis, 5 independent training runs each):

| Family | Model |
|---|---|
| `racnnclassiflows` | RACNN-ClassiFlow |
| `racnnflows` | RACNN-Flow |
| `racnnclass` | RACNN-Classifier |
| `cnnclassiflows` | CNN-ClassiFlow (no region conditioning) |
| `cnnflows` | CNN-Flow |
| `cnnclass` | CNN-Classifier |

**Checkpoints:**

| Checkpoint | Meaning |
|---|---|
| `best_model` | $\mathcal{L}$: lowest validation loss (ClassiFlow: NLL + BCE; Flow: NLL) |
| `best_model_ediff` | $\Delta$: entropy-difference criterion (mono-jet ClassiFlow and Flow only) |
| `best_model_c` | Classifier, lowest validation BCE |
| `best_model_f` | ClassiFlow, lowest validation NLL |

The paper's ensembles are the 3 runs per family with the lowest loss, listed in `ensembles.json` together with the bump-hunt single-best runs. The ensemble posterior is the equal-weight mixture of its members, and the ensemble classifier output is the mean logit.

## Usage

All models are float64 PyTorch modules. To build one and load its weights (from the repository root):

```python
import racnn_inference as ri
ens = ri.load_ensemble("bumphunt", "racnnclassiflows", "best_model", weights_dir="weights")
```

or by hand:

```python
import json, racnn_model as M      # bumphunt/ on sys.path
from safetensors.torch import load_file
cfg = json.load(open("weights/model_configs.json"))["bumphunt"]["racnnclassiflows"]
model = getattr(M, cfg["class"])(**cfg["kwargs"]).double()
model.load_state_dict(load_file("weights/bumphunt/racnnclassiflows_5/best_model.safetensors"))
```

Inputs must be normalised with the scalers stored in the dataset files.

## Checks

The weights were converted from the original PyTorch checkpoints, and every tensor is bit-identical. With the released datasets and code:

* the bump-hunt ensembles reproduce the paper's Table 2 test NLLs: RACNN-ClassiFlow −4.602, RACNN-Flow −4.490, CNN-ClassiFlow −3.004, CNN-Flow −2.986;
* the mono-jet RACNN-ClassiFlow run 5 ($\mathcal{L}$) and run 4 ($\Delta$) reproduce the Fig 15 starting points: −0.567 and −0.488.

## Licence

The weights are released under CC BY 4.0 (see `LICENSE` in this folder). The code of the repository is MIT-licensed.
