# racnn-meraao

Code and trained models for **"Multitask, Every Region, All at Once: Region-Conditioned Simulation-Based Inference with Signal Detection and Region Evaluation at Colliders"** (Y. S. Koay, T. Glüsenkamp, S. Moretti, P. Singh).

The paper introduces a region-aware framework: a shared backbone is conditioned on the signal-region (SR) definition $c$. One amortized model, trained once over many SRs and signal hypotheses, then serves three tasks for any candidate SR:

* **signal detection:** a classifier head gives $P(s+b \mid \mathcal{D}_c, c)$ and its logit $\Lambda$;
* **parameter inference:** a normalizing-flow head gives the posterior $p(\theta \mid \mathcal{D}_c, c)$;
* **SR ranking:** for a hypothesised signal, the logit $\Lambda$ and the posterior entropy $H$ rank candidate SRs, with no extra training.

The backbone is also a starting point that can be fine-tuned to one chosen SR with far less data than training from scratch.

The example model is **RACNN-ClassiFlow**, a region-aware CNN with a classifier head and a flow head. Its variants RACNN-Flow and RACNN-Classifier, and the region-agnostic CNN-* baselines are included as well; the NECNN-* baselines (paper appendix) come as code and run configurations only. The paper validates the approach on an idealized Gaussian bump hunt and on a mono-jet dark-matter search at the LHC.

| | |
|---|---|
| Datasets and figure data | [huggingface.co/datasets/kys-sheng/racnn-meraao](https://huggingface.co/datasets/kys-sheng/racnn-meraao) |
| Interactive bump-hunt demo | [huggingface.co/spaces/kys-sheng/racnn_bumphunt_demo](https://huggingface.co/spaces/kys-sheng/racnn_bumphunt_demo) |

## Contents

```
racnn_inference.py          helpers used by the example notebooks (load ensembles, posterior on a grid, NLL)
bumphunt/                   Gaussian bump hunt
  racnn_model.py              model classes: RACNN / CNN / NECNN x Flow / Classifier / ClassiFlow
  racnn_trainer.py            training and validation loops
  train_model.py              training entry point
  finetune_model.py           fine-tuning entry point (Fig 7)
  racnn_eval.py               evaluation utilities used for the paper
  flexibledataset.py          dataset class + HDF5 reader/writer
  bump_hunt.py                toy signal/background generator
  configs/                    the runs behind the paper (one line = one run)
  inference_plots.py          the inference figure of the paper (Fig 3)
  inference_example.ipynb
monojet/                    mono-jet dark-matter search (same layout, *_mx.py; inference_plots.py: Figs 10, 11)
weights/                    trained weights of the paper's models (safetensors, CC BY 4.0)
scripts/download.py         fetches the datasets into data/ from Hugging Face
```

## Installation

```bash
git clone https://github.com/kys-sheng/racnn-meraao.git
cd racnn-meraao
pip install -r requirements.txt
python scripts/download.py            # datasets, ~1.3 GB; --no-train for the notebooks only (~0.7 GB)
```

The normalizing flows come from [jammy_flows](https://github.com/thoglu/jammy_flows), installed from GitHub at the commit the models were trained with. It needs `healpy`, which has no Windows wheels, so use Linux, macOS or WSL. Everything runs on a CPU. A GPU is used automatically when one is available.

## Quick start

Open `bumphunt/inference_example.ipynb` or `monojet/inference_example.ipynb`. They show the paper's inference examples (Fig 3; Figs 10 and 11) with the released RACNN-ClassiFlow ensemble:

* the histograms in the SR: signal + background, background and signal;
* the ensemble posterior on the signal-plus-background and on the background-only ensemble;
* the classifier output $P(s+b \mid \mathcal{D}_c, c)$, the logit $\Lambda$ and the posterior entropy $H$.

The bump-hunt notebook simulates new examples with `bump_hunt.py`, so any θ and window can be tried. The mono-jet notebook reads the examples from the test set. Both run on a CPU in about a minute.

In your own code:

```python
import racnn_inference as ri
_, fd = ri.import_analysis("bumphunt")                 # puts bumphunt/ on sys.path
test = fd.load_h5("data/bumphunt/test.h5")             # a torch Dataset: (x_hist, x_aux, y_flow, y_mlp)
ens  = ri.load_ensemble("bumphunt", "racnnclassiflows", "best_model", "weights")
```

## Data and weights

**Weights** (in `weights/`, see its README). For each analysis there are 6 model families (RACNN/CNN × ClassiFlow/Flow/Classifier) with 5 training runs each, as `safetensors`.
* `weights/model_configs.json` holds the constructor arguments.
* `weights/ensembles.json` lists the 3 runs that form each paper ensemble.
* Checkpoints: `best_model` is the lowest-validation-loss checkpoint ($\mathcal{L}$); mono-jet ClassiFlow and Flow also have `best_model_ediff` ($\Delta$); the classifiers use `best_model_c`.

**Datasets** (on Hugging Face, downloaded into `data/`). HDF5 files with the train/validation/test splits, the fine-tuning sets of Figs 7 and 15, and the mono-jet parameter-point sets (`theta_sets.json`). Arrays are stored normalised, exactly as the networks see them, together with the scalers. `ds.unnormalize_input(...)` and `ds.unnormalize_output(...)` return physical values.

**Figure data.** CSV files for Figs 7, 15 and 16. Column descriptions are in the dataset card.

## Training and fine-tuning

The config files record how every model in the paper was trained. Each line is one run, written as `<script> <arguments>`. Run a line from the analysis folder, for example:

```bash
cd bumphunt
python $(grep -v '^#' configs/main_models.txt | sed -n 1p)        # RACNN-ClassiFlow, run 1
python $(grep -v '^#' configs/finetuning_fig7.txt | sed -n 1p)    # Fig 7: fine-tuning, low SR, 1 dataset per theta
```

| Config | Runs | What it trains |
|---|---|---|
| `bumphunt/configs/main_models.txt` | 45 | the bump-hunt models: RACNN / CNN × ClassiFlow / Flow / Classifier, and the NECNN baseline |
| `bumphunt/configs/finetuning_fig7.txt` | 200 | Fig 7: RACNN-ClassiFlow fine-tuned (flow head only), and RACNN-ClassiFlow and CNN-Flow trained from scratch |
| `monojet/configs/main_models.txt` | 45 | the mono-jet models, as above |
| `monojet/configs/finetuning_fig15.txt` | 180 | Fig 15: RACNN-ClassiFlow-$\mathcal{L}$ fine-tuned (linear or log normalisation, flow head only or full), and RACNN-ClassiFlow and CNN-Flow trained from scratch |

Training writes to `runs/`. Run-to-run randomness means a retrained model will not be bit-identical to the released weights.

## About this repository

This repository was prepared with the help of [Claude Code](https://claude.com/claude-code), starting from the authors' original research code and working directories. Claude Code:
* collected the code used for the paper and removed cluster-specific paths;
* added HDF5 and safetensors loading;
* converted the datasets to HDF5 and the weights to safetensors;
* wrote the example notebooks and this documentation.

The model, training and evaluation code is otherwise the authors' original.

## Licence

The code is released under the MIT licence (see `LICENSE`). The weights (`weights/LICENSE`) and the datasets on Hugging Face are released under CC BY 4.0.

## Citation

The BibTeX entry will be added once the paper is on arXiv.
