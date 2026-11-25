"""Mono-jet inference figures (paper Figs 10 and 11): input histograms of one signal region
(signal+background, background, signal) and the ensemble posterior for the
signal-plus-background and the background-only ensemble.
"""
import os
import sys

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import racnn_inference as ri  # noqa: E402

OBS = ("PT", "MET", "ETA")
OBS_LATEX = {"PT": (r"$p_T$", r"$p_T$ (GeV)"), "MET": (r"$E\!\!\!\!/_T$", r"$E\!\!\!\!/_T$ (GeV)"), "ETA": (r"$\eta$", r"$\eta$")}
SR_KEYS = ("PT_CUT", "ETA_CUT", "PT_LOW", "PT_UP", "MET_LOW", "MET_UP", "ETA_LOW", "ETA_UP")


def find_example(path, mass, coup, sr):
    """Rows of one test file for theta = (mass, coup) and signal region sr (dict of SR_KEYS, physical units).

    Returns (dataset with the signal-plus-background row first and the matching background-only row
    second, its row numbers in the file). The background-only row is chosen as in the paper notebook:
    same signal region, every bin at most the signal-plus-background one, and the closest event-count
    difference to the number of signal events.
    """
    import flexibledataset as fd
    sc = fd.load_h5(path, index=[0])
    with h5py.File(path) as f:
        un = lambda grp, k, side: sc._unnormalize(f[f"{grp}/{k}"][()], (sc.input_scalers if side == "in" else sc.output_scalers)[k])
        m, g = un("flow_outputs", "MASS", "out"), un("flow_outputs", "COUP", "out")
        in_sr = np.ones(len(m), bool)
        for k in SR_KEYS:
            in_sr &= np.isclose(un("other_inputs", k, "in"), sr[k], atol=1e-3 * max(1.0, abs(sr[k])))
    sb = np.where(in_sr & np.isclose(m, mass, atol=0.5) & np.isclose(g, coup, atol=1e-3))[0]
    bg = np.where(in_sr & np.isnan(m))[0]
    if len(sb) == 0 or len(bg) == 0:
        return None, None
    rows = np.concatenate([sb[:1], bg])
    d = fd.load_h5(path, index=rows)
    h = {o: d.unnormalize_input(o, d.histograms[o]) for o in OBS}
    n = {o: d.unnormalize_input(f"{o}_NEVENTS", d.other_inputs[f"{o}_NEVENTS"]) for o in OBS}
    nsig = {o: d.mlp_outputs[f"{o}_NEVENTS_SIG"][0] for o in OBS}
    valid = np.all([np.all(h[o][1:] <= h[o][0] + 1e-9, axis=1) for o in OBS], axis=0)
    cost = sum(np.abs((n[o][0] - n[o][1:]) - nsig[o]) for o in OBS)
    cand = np.where(valid)[0] if valid.any() else np.arange(len(bg))
    sel = rows[[0, int(cand[np.argmin(cost[cand])]) + 1]]          # (s+b row, background-only row)
    order = np.argsort(sel)                                        # load_h5 needs increasing row numbers
    d = fd.load_h5(path, index=sel[order])
    return (_swap(d) if order[0] == 1 else d), sel


def _swap(d):
    """Reorder a two-row dataset so the signal-plus-background row comes first."""
    for grp in ("histograms", "other_inputs", "flow_outputs", "mlp_outputs"):
        dd = getattr(d, grp)
        for k in dd:
            dd[k] = dd[k][::-1].copy()
    return d


def plot_data(pair, sr, mass, coup, yscale="log"):
    """Input histograms of the signal region: signal+background, background-only, and their difference (signal)."""
    FS = 17
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, o in zip(axes, OBS):
        lab, xlab = OBS_LATEX[o]
        sb, b = (pair.unnormalize_input(o, pair.histograms[o][i]) for i in (0, 1))
        s = sb - b
        x = np.linspace(sr[f"{o}_LOW"], sr[f"{o}_UP"], len(sb))
        bw = x[1] - x[0]
        ax.bar(x, sb, width=bw, alpha=0.3, color="blue", label=f"Sig+Bg (N={sb.sum():.0f})")
        ax.bar(x, b, width=bw, alpha=0.5, color="orange", label=f"Background (N={b.sum():.0f})")
        ax.bar(x, s, width=bw, alpha=0.7, color="green", label=f"Signal (N={s.sum():.0f})")
        ax.step(x, sb, where="mid", color="blue", lw=1)
        ax.set(xlim=(sr[f"{o}_LOW"], sr[f"{o}_UP"]), yscale=yscale)
        ax.set_xlabel(xlab, fontsize=FS - 1)
        ax.set_ylabel("Number of Events", fontsize=FS)
        lo, hi = sr[f"{o}_LOW"], sr[f"{o}_UP"]
        cut = sr["ETA_CUT"] if o == "ETA" else sr["PT_CUT"]
        rng = f"[{lo:.1f},{hi:.1f}]" if o == "ETA" else f"[{lo:.0f},{hi:.0f}]"
        ax.set_title(f"{lab} : Range={rng}, Cut$>${cut:.1f}" if o == "ETA" else f"{lab} : Range={rng}, Cut$>${cut:.0f}", fontsize=FS)
        ax.legend(fontsize=FS - 2)
        ax.tick_params(labelsize=FS - 2)
    fig.suptitle(rf"$m_{{\rm DM}}=${mass:g} GeV, $g={coup:g}$", fontsize=FS)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_posteriors(models, pair, mass, coup, grid_bins=100, share_colorbar=False, device="cpu"):
    """Ensemble posterior for the signal-plus-background row (left) and the background-only row (right),
    with the 68% credible contour, the classifier output P(s+b | D_c, c), the logit Lambda and the entropy H."""
    FS = 17
    sc = pair.output_scalers
    rng = ((sc["MASS"]["min"], sc["MASS"]["max"]), (sc["COUP"]["min"], sc["COUP"]["max"]))
    panels = []
    for i in (0, 1):
        x_cnn, x_aux, _, _ = (t.unsqueeze(0).double().to(device) for t in pair[i])
        g0, g1, dens, H = ri.posterior_grid(models, pair, "monojet", x_cnn, x_aux, *rng, bins=grid_bins)
        with torch.no_grad():
            logits = [float(m(x_cnn, x_aux)[1].mean()) for m in models]
        panels.append((g0, g1, dens, H, float(np.mean([1 / (1 + np.exp(-l)) for l in logits])), float(np.mean(logits))))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=share_colorbar, constrained_layout=True)
    vmax = max(p[2].max() for p in panels)
    titles = (r"Posterior $q(m,g|D_c,c)$  (Y=s+b)", r"Posterior $q(m,g|D_c,c)$  (Y=b)")
    ims = []
    for i, (ax, (g0, g1, dens, H, p, lam)) in enumerate(zip(axes, panels)):
        im = ax.pcolormesh(g0, g1, dens, shading="auto", cmap="viridis", **({"vmin": 0, "vmax": vmax} if share_colorbar else {}))
        ims.append(im)
        bin_area = (rng[0][1] - rng[0][0]) / dens.shape[0] * (rng[1][1] - rng[1][0]) / dens.shape[1]
        flat = np.sort(dens.ravel())[::-1]
        cum = np.cumsum(flat * bin_area)
        if cum[-1] > 0:
            lvl = flat[min(np.searchsorted(cum / cum[-1], 0.6827), len(flat) - 1)]
            cs = ax.contour(g0, g1, dens, levels=[lvl], colors="white", alpha=0.8, linewidths=1.2, linestyles=[":"])
            ax.clabel(cs, fmt={lvl: "68%"})
        if i == 0:
            ax.scatter(mass, coup, c="r", s=100, marker="x", label="Truth", zorder=3)
        ax.scatter([], [], marker="", label=rf"$P(s\!+\!b|D_c,c) = {p:.3f}$")
        ax.scatter([], [], marker="", label=rf"$\Lambda = {lam:.2f}$")
        ax.scatter([], [], marker="", label=f"$H = {H:.2f}$")
        ax.legend(fontsize=FS - 2)
        ax.set_xlabel(r"Mass, $m_{\rm DM}$ (GeV)", fontsize=FS - 1)
        if i == 0 or not share_colorbar:
            ax.set_ylabel(r"Coupling, $g$", fontsize=FS - 1)
        ax.set_title(titles[i], fontsize=FS)
        ax.tick_params(labelsize=FS - 2)
        if not share_colorbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Probability Density", fontsize=FS - 1)
    if share_colorbar:
        cb = fig.colorbar(ims[0], ax=list(axes), fraction=0.046, pad=0.04)
        cb.set_label("Probability Density", fontsize=FS - 1)
    return fig
