"""Inference helpers shared by the bump-hunt and mono-jet example notebooks.

Everything here works on the released files:
  weights/<analysis>/<family>_<run>/<checkpoint>.safetensors
  weights/model_configs.json, weights/ensembles.json
  data/<analysis>/*.h5
and on the model code in bumphunt/ or monojet/.
"""
import importlib
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

ANALYSES = {
    # code folder, flow targets (posterior parameters), signal-count outputs
    "bumphunt": dict(code="bumphunt", targets=("mean", "sigma"), signal_keys=("nevents_sg",), signal_threshold=0.0),
    "monojet":  dict(code="monojet",  targets=("MASS", "COUP"),  signal_keys=("PT_NEVENTS_SIG", "MET_NEVENTS_SIG", "ETA_NEVENTS_SIG"), signal_threshold=1.0),
}
REPO = os.path.dirname(os.path.abspath(__file__))


def import_analysis(analysis):
    """Put the analysis code folder on sys.path and return (model module, flexibledataset module)."""
    code_dir = os.path.join(REPO, ANALYSES[analysis]["code"])
    for other in ANALYSES.values():  # the two folders share module names
        p = os.path.join(REPO, other["code"])
        if p in sys.path and p != code_dir:
            sys.path.remove(p)
    for name in ("flexibledataset", "racnn_model", "racnn_model_mx"):
        sys.modules.pop(name, None)
    sys.path.insert(0, code_dir)
    cfg_module = "racnn_model" if analysis == "bumphunt" else "racnn_model_mx"
    return importlib.import_module(cfg_module), importlib.import_module("flexibledataset")


def load_model(analysis, family, run, checkpoint, weights_dir, device="cpu"):
    """Build one model from model_configs.json and load its weights."""
    model_mod, _ = import_analysis(analysis)
    cfg = json.load(open(os.path.join(weights_dir, "model_configs.json")))[analysis][family]
    model = getattr(model_mod, cfg["class"])(**cfg["kwargs"]).double()
    path = os.path.join(weights_dir, analysis, f"{family}_{run}", f"{checkpoint}.safetensors")
    model.load_state_dict(load_file(path))
    model.cnn_mode = cfg["cnn_mode"]
    return model.to(device).eval()


def load_ensemble(analysis, family, checkpoint, weights_dir, device="cpu", runs=None):
    """The paper's ensemble (3 best of 5 runs, from ensembles.json) unless `runs` is given."""
    if runs is None:
        runs = json.load(open(os.path.join(weights_dir, "ensembles.json")))[analysis][family][checkpoint]
    return [load_model(analysis, family, r, checkpoint, weights_dir, device) for r in runs]


def _forward(model, x_cnn, x_aux):
    return model(x_cnn) if model.cnn_mode else model(x_cnn, x_aux)


def conditionals(model, x_cnn, x_aux):
    """Flow conditioning vector and classifier logit (None where the model has no such head)."""
    out = _forward(model, x_cnn, x_aux)
    if isinstance(out, tuple):
        return out[0], out[1]
    return (out, None) if hasattr(model, "pdf") else (None, out)


def is_signal(y_mlp, analysis):
    """True where signal is present (same definition as the paper's evaluation)."""
    return (y_mlp >= ANALYSES[analysis]["signal_threshold"]).any(dim=1) if analysis == "monojet" else (y_mlp.squeeze(-1) > 0)


@torch.no_grad()
def classifier_probability(models, x_cnn, x_aux):
    """Ensemble-mean classifier logit and the sigmoid of it, per input row."""
    logits = torch.stack([conditionals(m, x_cnn, x_aux)[1].squeeze(-1) for m in models])
    mean_logit = logits.mean(0)
    return mean_logit, torch.sigmoid(mean_logit)


@torch.no_grad()
def ensemble_log_prob(models, x_cnn, x_aux, y_norm):
    """log of the ensemble posterior density (equal-weight mixture of members) at normalised targets y_norm."""
    lps = []
    for m in models:
        cond, _ = conditionals(m, x_cnn, x_aux)
        lp, _, _ = m.pdf(y_norm, conditional_input=cond)
        lps.append(torch.nan_to_num(lp, nan=-1e10))
    return torch.logsumexp(torch.stack(lps), 0) - np.log(len(models))


def _scale(ds, key):
    s = ds.output_scalers.get(key, {"method": "none"})
    if s["method"] == "standard":
        return float(s["std"])
    if s["method"] in ("local_minmax", "global_minmax"):
        return float(s["max"] - s["min"])
    return 1.0


@torch.no_grad()
def posterior_grid(models, ds, analysis, x_cnn, x_aux, range0, range1, bins=80, batch=20000):
    """Ensemble posterior for ONE input (x_cnn, x_aux with batch dim 1) on a grid in physical units.

    ds: a dataset carrying the output scalers (any split of the same analysis).
    Returns (grid0, grid1, density in physical units, entropy in normalised units, as in the paper).
    """
    k0, k1 = ANALYSES[analysis]["targets"]
    g0, g1 = np.meshgrid(np.linspace(*range0, bins), np.linspace(*range1, bins), indexing="ij")
    norm = ds.normalize_batch("flow_outputs", {k0: g0.ravel(), k1: g1.ravel()})
    y = torch.tensor(np.stack([norm[k0], norm[k1]], 1), dtype=torch.float64, device=x_cnn.device)
    dens = []
    for s in range(0, len(y), batch):
        yb = y[s:s + batch]
        dens.append(ensemble_log_prob(models, x_cnn.expand(len(yb), *x_cnn.shape[1:]), x_aux.expand(len(yb), *x_aux.shape[1:]), yb).exp().cpu())
    p_norm = torch.cat(dens).numpy()
    jac = 1.0 / (_scale(ds, k0) * _scale(ds, k1))
    dA_norm = (range0[1] - range0[0]) / bins * (range1[1] - range1[0]) / bins * jac
    pos = p_norm > 0
    entropy = float(-np.sum(p_norm[pos] * np.log(p_norm[pos]) * dA_norm))
    return g0, g1, (p_norm * jac).reshape(bins, bins), entropy


@torch.no_grad()
def mean_nll(models, loader, analysis, device="cpu"):
    """Mean -log p(theta_true | x) over signal-present rows (ensemble mixture if several models)."""
    out = []
    for x_cnn, x_aux, y_flow, y_mlp in loader:
        x_cnn, x_aux, y_flow, y_mlp = (t.double().to(device) for t in (x_cnn, x_aux, y_flow, y_mlp))
        s = is_signal(y_mlp, analysis)
        if s.any():
            out.append(-ensemble_log_prob(models, x_cnn[s], x_aux[s], y_flow[s]).cpu())
    return float(torch.cat(out).mean())
