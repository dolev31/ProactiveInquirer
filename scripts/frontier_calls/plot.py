"""The calls-axis figure, beside the cap-axis one, per suite.

Style borrowed from `src/pi_run/render.py` (Agg backend, the same rcParams) so this sits next
to `pi render`'s own figures without a seam, though it is not one of F1-F3 and does not go
through `pi render`'s provenance machinery -- its own provenance is `data.json`, written by
`compute.py` in the same run.
"""

from __future__ import annotations

from pathlib import Path


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    return plt


_PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")  # Okabe-Ito, matches render.py's use


def plot_suite(suite_name: str, cells: dict, out_dir: Path) -> Path:
    plt = _mpl()
    arms = sorted({a for a, _c in cells})
    caps = sorted({c for _a, c in cells})
    metric = next(iter(cells.values())).metric_name

    fig, (ax_cap, ax_calls) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for i, arm in enumerate(arms):
        color = _PALETTE[i % len(_PALETTE)]
        ys = [cells[(arm, c)].metric_point for c in caps]
        lo = [cells[(arm, c)].metric_ci_lo for c in caps]
        hi = [cells[(arm, c)].metric_ci_hi for c in caps]
        xs_calls = [cells[(arm, c)].mean_retrieval_calls for c in caps]

        ax_cap.plot(caps, ys, marker="o", markersize=4, color=color, label=arm, linewidth=1.6)
        ax_cap.fill_between(caps, lo, hi, color=color, alpha=0.15, linewidth=0.0)

        order = sorted(range(len(xs_calls)), key=lambda k: xs_calls[k])
        xs_sorted = [xs_calls[k] for k in order]
        ys_sorted = [ys[k] for k in order]
        lo_sorted = [lo[k] for k in order]
        hi_sorted = [hi[k] for k in order]
        ax_calls.plot(
            xs_sorted, ys_sorted, marker="o", markersize=4, color=color, label=arm, linewidth=1.6
        )
        ax_calls.fill_between(
            xs_sorted, lo_sorted, hi_sorted, color=color, alpha=0.15, linewidth=0.0
        )

    ax_cap.set_xlabel("budget cap (allowance, not spend)")
    ax_cap.set_ylabel(metric)
    ax_cap.set_title(f"{suite_name}: cap-indexed")
    ax_cap.set_xticks(caps)
    ax_cap.legend(loc="best", frameon=True, framealpha=0.85, edgecolor="none", fontsize=7)

    ax_calls.set_xlabel("mean realized retrieval calls (spend)")
    ax_calls.set_ylabel(metric)
    ax_calls.set_title(f"{suite_name}: calls-indexed")
    ax_calls.legend(loc="best", frameon=True, framealpha=0.85, edgecolor="none", fontsize=7)

    fig.suptitle(
        f"{suite_name}: the same 5 cells, cap-indexed (left, an allowance) vs "
        "calls-indexed (right, the realized spend)",
        fontsize=8,
    )
    fig.tight_layout()
    pdf = out_dir / f"{suite_name}_calls_axis.pdf"
    png = out_dir / f"{suite_name}_calls_axis.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    return pdf


def plot_all(cells_by_suite: dict, out_dir: Path) -> list[Path]:
    # NOT "figures/": .gitignore's unanchored `figures/` rule (it exists to keep `pi render`'s
    # own regenerable output untracked) matches a directory of that name ANYWHERE in the tree,
    # so a subdirectory literally called "figures" here would be silently excluded from every
    # `git add` -- confirmed with `git check-ignore -v` before this name was chosen.
    figs_dir = out_dir / "calls_axis_figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    return [plot_suite(name, cells, figs_dir) for name, cells in cells_by_suite.items()]
