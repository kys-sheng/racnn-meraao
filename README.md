# racnn-meraao

Code for **"Multitask, Every Region, All at Once: Adaptive Region-Conditioned SBI for Signal Detection, Parameter Inference and Signal-Region Ranking at the colliders"** (Y. S. Koay, T. Glüsenkamp, S. Moretti, P. Singh).

A region-aware CNN (RACNN) conditions a shared backbone on the signal-region (SR) definition. One amortized model then gives, for any SR and any signal hypothesis:

* a classifier probability that signal is present (signal detection),
* a normalizing-flow posterior over the signal parameters (parameter inference),
* the posterior entropy, used to rank SRs.

The backbone can also be fine-tuned to one fixed SR with little data. The paper validates this on an idealized Gaussian bump hunt and on a mono-jet dark-matter search at the LHC.

| | |
|---|---|
| Trained weights | [huggingface.co/kys-sheng/racnn-meraao](https://huggingface.co/kys-sheng/racnn-meraao) |
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
  inference_example.ipynb
monojet/                    mono-jet dark-matter search (same layout, *_mx.py)
scripts/download.py         fetches weights/ and data/ from Hugging Face
```

## Installation

```bash
git clone https://github.com/kys-sheng/racnn-meraao.git
cd racnn-meraao
pip install -r requirements.txt
python scripts/download.py            # ~1.4 GB; see --help for smaller subsets
```

The normalizing flows come from [jammy_flows](https://github.com/thoglu/jammy_flows), installed from GitHub at the commit the models were trained with. It needs `healpy`, which has no Windows wheels, so use Linux, macOS or WSL. Everything runs on a CPU. A GPU is used automatically when one is available.

## Quick start

Open `bumphunt/inference_example.ipynb` or `monojet/inference_example.ipynb`. Each notebook:

* loads the paper's RACNN-ClassiFlow ensemble and part of the test set,
* plots the input histograms, the ensemble posterior and the signal probability for example SRs,
* checks a number against the paper: the bump-hunt notebook reproduces the Table 2 ensemble NLLs, the mono-jet notebook the Fig 15 starting point.

In your own code:

```python
import racnn_inference as ri
_, fd = ri.import_analysis("bumphunt")                 # puts bumphunt/ on sys.path
test = fd.load_h5("data/bumphunt/test.h5")             # a torch Dataset: (x_hist, x_aux, y_flow, y_mlp)
ens  = ri.load_ensemble("bumphunt", "racnnclassiflows", "best_model", "weights")
```

## Data and weights

**Weights.** For each analysis there are 6 model families (RACNN/CNN × ClassiFlow/Flow/Classifier) with 5 training runs each, as `safetensors`.
* `weights/model_configs.json` holds the constructor arguments.
* `weights/ensembles.json` lists the 3 runs that form each paper ensemble.
* Checkpoints: `best_model` is the lowest-validation-loss checkpoint ($\mathcal{L}$); mono-jet ClassiFlow and Flow also have `best_model_ediff` ($\Delta$); the classifiers use `best_model_c`.

**Datasets.** HDF5 files with the train/validation/test splits, the fine-tuning sets of Figs 7 and 15, and the mono-jet parameter-point sets (`theta_sets.json`). Arrays are stored normalised, exactly as the networks see them, together with the scalers. `ds.unnormalize_input(...)` and `ds.unnormalize_output(...)` return physical values.

**Figure data.** CSV files for Figs 7, 15 and 16. Column descriptions are in the dataset card.

## Training and fine-tuning

Each line of a config file is the argument list of one run. Run them from the analysis folder, for example:

```bash
cd bumphunt
python train_model.py $(grep -v '^#' configs/paper_models.txt | sed -n 1p)       # RACNN-ClassiFlow, run 1
python finetune_model.py $(grep -v '^#' configs/finetune_fig7.txt | sed -n 1p)   # Fig 7, low SR, N = 1
```

| Config | What it trains |
|---|---|
| `bumphunt/configs/paper_models.txt` | the bump-hunt models, incl. the NECNN baseline |
| `bumphunt/configs/finetune_fig7.txt` | Fig 7 fine-tuning |
| `bumphunt/configs/scratch_fig7.txt` | Fig 7 from-scratch CNN-Flow baselines |
| `bumphunt/configs/scratch_racnn_fig7.txt` | Fig 7 from-scratch RACNN-ClassiFlow baselines |
| `monojet/configs/paper_models.txt` | the mono-jet models, incl. the NECNN baseline |
| `monojet/configs/finetune_fig15.txt` | Fig 15 fine-tuning: linear or log normalisation, flow head only or full |
| `monojet/configs/scratch_fig15.txt` | Fig 15 from-scratch RACNN-ClassiFlow and CNN-Flow baselines |

Training writes to `runs/`. Run-to-run randomness means a retrained model will not be bit-identical to the released weights.

## Reproducibility checks

With only the released files and this code:

* **Bump hunt.** For each model family, the released weights of the paper's best run reproduce the validation loss stored at the saved epoch, to machine precision. The ensemble test NLLs of Table 2 are reproduced: RACNN-ClassiFlow −4.602, RACNN-Flow −4.490, CNN-ClassiFlow −3.004, CNN-Flow −2.986.
* **Mono-jet.** The Fig 15 starting points are reproduced: RACNN-ClassiFlow-$\mathcal{L}$ run 5, NLL −0.567; $\Delta$ run 4, NLL −0.488. The classifiers reproduce their stored validation losses. The flow validation losses logged during the original training differ from what this version of the trainer computes for the same weights, so retraining may select different checkpoints than the released ones.

## Licence

The code is released under the MIT licence (see `LICENSE`). The weights and datasets on Hugging Face are released under CC BY 4.0.

## Citation

The BibTeX entry will be added once the paper is on arXiv.
