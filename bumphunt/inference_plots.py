"""Bump-hunt inference figure (paper Fig 3): simulated input histograms and the
exact ensemble posterior for the signal-plus-background and background-only
ensembles, with the classifier output and the posterior entropy H.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

import bump_hunt as bh


def _calculate_confidence_levels(density_map, bin_area, target_fractions=(0.6827, 0.9545, 0.9973)):
    """Density thresholds enclosing the target probability-mass fractions."""
    sorted_pdf = np.sort(density_map.flatten())[::-1]
    cum_mass = np.cumsum(sorted_pdf * bin_area)
    if cum_mass[-1] == 0:
        return [0] * len(target_fractions)
    norm_mass = cum_mass / cum_mass[-1]
    levels = [sorted_pdf[min(np.searchsorted(norm_mass, f), len(sorted_pdf) - 1)] for f in target_fractions]
    return sorted(levels)


def get_exact_ensemble_pdf_grid(model_list, x_cnn, x_aux, mu_range, sigma_range, flexidataset,
                                grid_bins=100, batch_size=10000, device="cpu"):
    """Ensemble (equal-weight mixture) posterior on a (theta_m, theta_s) grid.

    Returns (Mu_grid, Sigma_grid, density in physical units, entropy H in the
    normalised parameter space used throughout the paper).
    """
    for m in model_list:
        m.eval()
    gb_mu, gb_sigma = (grid_bins, grid_bins) if isinstance(grid_bins, int) else grid_bins
    Mu_grid, Sigma_grid = np.meshgrid(np.linspace(*mu_range, gb_mu), np.linspace(*sigma_range, gb_sigma), indexing="ij")
    norm = flexidataset.normalize_batch("flow_outputs", {"mean": Mu_grid.ravel(), "sigma": Sigma_grid.ravel()})
    Y_norm = torch.vstack([torch.tensor(norm["mean"]), torch.tensor(norm["sigma"])]).T.to(device)

    with torch.no_grad():
        conds = []
        for m in model_list:
            out = m(x_cnn, x_aux)
            conds.append((out[0] if isinstance(out, tuple) else out).double().to(device))
        pdf = np.zeros(len(Y_norm))
        for s in range(0, len(Y_norm), batch_size):
            y = Y_norm[s:s + batch_size].double()
            acc = torch.zeros(len(y), device=device, dtype=torch.float64)
            for m, c in zip(model_list, conds):
                lp, _, _ = m.pdf.forward(y, conditional_input=c.repeat(len(y), 1))
                acc += torch.exp(torch.nan_to_num(lp, nan=-1e10).squeeze())
            pdf[s:s + len(y)] = (acc / len(model_list)).cpu().numpy()

    def _scale(key):
        sc = flexidataset.output_scalers.get(key, {"method": "none"})
        if sc.get("method") == "standard":
            return sc["std"]
        if sc.get("method") in ("local_minmax", "global_minmax"):
            return sc["max"] - sc["min"]
        return 1.0
    jacobian = 1.0 / (_scale("mean") * _scale("sigma"))
    dA_norm = (mu_range[1] - mu_range[0]) / gb_mu * (sigma_range[1] - sigma_range[0]) / gb_sigma * jacobian
    pos = pdf > 0
    H = float(-np.sum(pdf[pos] * np.log(pdf[pos]) * dA_norm))
    return Mu_grid, Sigma_grid, (pdf * jacobian).reshape(gb_mu, gb_sigma), H


def generate_exact_inference_plot_paper_ready(mean, sigma, lower_limit, upper_limit, model_list, flexidataset,
                                              grid_bins=100, mu_range=(2, 28), sigma_range=(1, 5),
                                              share_colorbar=True, share_y=True, device="cpu", save_path=None):
    """Simulate one dataset at theta = (mean, sigma), select the window [lower_limit, upper_limit],
    and plot the input histograms and the ensemble posterior for the signal-plus-background and the
    background-only ensemble. The legend gives the classifier output P(s+b | D_c, c) (mean of the
    members' sigmoid outputs), the ensemble-mean logit Lambda and the posterior entropy H.

    flexidataset: any bump-hunt dataset (its scalers are used to normalise the inputs).
    Uses numpy's global random state: call np.random.seed(...) first for a reproducible example.
    """
    FS = 17
    sg, bg = bh.generate_data(mean, sigma)
    hist_sg, hist_bg, hist_both = bh.ensemble_data(sg, bg, yscale="linear", xrange=[lower_limit, upper_limit])
    neve, neve_bg, neve_sg = sum(hist_both), sum(hist_bg), sum(hist_sg)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.08, wspace=0.05)
    x = np.linspace(lower_limit, upper_limit, bh.NBINS)
    bw = x[1] - x[0]
    ax1.bar(x, hist_both, label=f"Sig+Bg (N={neve:.0f})", width=bw, alpha=.5)
    ax1.bar(x, hist_bg, label=f"Background (N={neve_bg:.0f})", width=bw, alpha=.5)
    ax1.bar(x, hist_sg, label=f"Signal (N={neve_sg:.0f})", width=bw, alpha=.5)
    ax1.set_xlim(min(x), max(x))
    ax1.set_title(f"Input Histogram at SR=[{lower_limit},{upper_limit}]", fontsize=FS)
    ax1.set_xlabel("Observable, $x$", fontsize=FS - 1)
    ax1.set_ylabel("Number of Events", fontsize=FS)
    ax1.legend(fontsize=FS)
    ax1.tick_params(labelsize=FS - 2)

    def prep(hist, n_events):
        cnn = flexidataset.normalize_batch("histograms", {"histogram": hist})
        aux = flexidataset.normalize_batch("other_inputs", {"lower_limit": lower_limit, "upper_limit": upper_limit, "nevents": n_events})
        x_c = torch.from_numpy(np.array(list(cnn.values()), dtype=np.float64)).unsqueeze(0).to(device)
        x_a = torch.from_numpy(np.array(list(aux.values()), dtype=np.float64)).unsqueeze(0).to(device)
        return x_c, x_a

    panels = []
    for (x_c, x_a), ax, ttl in ((prep(hist_both, neve), ax2, "Signal+Background"), (prep(hist_bg, neve_bg), ax3, "Background Only")):
        with torch.no_grad():
            logits = [float(m(x_c, x_a)[1].mean()) for m in model_list]
        p_c = float(np.mean([1 / (1 + np.exp(-l)) for l in logits]))
        Mu, Sigma, Pdfs, H = get_exact_ensemble_pdf_grid(model_list, x_c, x_a, mu_range, sigma_range, flexidataset, grid_bins, device=device)
        panels.append((Mu, Sigma, Pdfs, ax, ttl, p_c, float(np.mean(logits)), H))
    vmax = max(float(p[2].max()) for p in panels)

    im = None
    for Mu, Sigma, Pdfs, ax, ttl, p_c, lam, H in panels:
        if share_colorbar:
            im = ax.pcolormesh(Mu, Sigma, Pdfs, shading="auto", cmap="viridis", vmin=0, vmax=vmax)
        else:
            im = ax.pcolormesh(Mu, Sigma, Pdfs, shading="auto", cmap="viridis")
            cb = fig.colorbar(im, ax=ax)
            cb.set_label("Probability Density", fontsize=FS - 1)
            cb.ax.tick_params(labelsize=FS - 2)
        bin_area = (mu_range[1] - mu_range[0]) / Pdfs.shape[0] * (sigma_range[1] - sigma_range[0]) / Pdfs.shape[1]
        ax.contour(Mu, Sigma, Pdfs, levels=_calculate_confidence_levels(Pdfs, bin_area, [0.6827]),
                   colors="white", alpha=0.8, linewidths=1.2, linestyles=[":"])
        ax.grid(False)
        ax.set_xlabel(r"Signal Position, $\theta_m$", fontsize=FS - 1)
        ax.set_title(rf"$P(\theta_m,\theta_s|D_c,c)$  ({ttl})", fontsize=FS)
        ax.tick_params(labelsize=FS - 2)
        if share_y and ax is ax3:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r"Signal Width, $\theta_s$", fontsize=FS - 1)
        ax.scatter([], [], marker="", label=f"$P(s+b|D_c,c) = {p_c:.3f}$")
        ax.scatter([], [], marker="", label=rf"$\Lambda = {lam:.2f}$")
        ax.scatter([], [], marker="", label=f"$H = {H:.2f}$")
        ax.legend(fontsize=FS - 2)
    ax2.scatter(mean, sigma, c="r", marker="x", s=100, label="Truth", zorder=3)
    ax2.legend(fontsize=FS - 2)
    if share_colorbar and im is not None:
        cb = fig.colorbar(im, ax=[ax2, ax3], fraction=0.05, pad=0.02)
        cb.set_label("Probability Density", fontsize=FS - 1)
        cb.ax.tick_params(labelsize=FS - 2)
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
