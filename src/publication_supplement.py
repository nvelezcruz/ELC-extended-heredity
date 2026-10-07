from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from string import Template
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import zipfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

from .corrected_pilot_model import (
    COMPONENTS,
    PilotConfig,
    _params,
    apply_parameter_overrides,
    level_m,
    level_steps,
    nonzero_cross_level_edges,
)
from .final_conditional_dependence_reconstruction import (
    GraphReconstructionConfig,
    known_dependency_table,
    reclassify_graph_dependencies,
    run_final_conditional_dependence_reconstruction,
)


FIG_DPI = 450
PALETTE = {
    "development": "#0072B2",
    "microbiome": "#009E73",
    "life_history": "#D55E00",
    "epigenetic": "#CC79A7",
    "ecological": "#E69F00",
    "background": "#666666",
    "null": "#8F8F8F",
}

PRIMARY_CORRECTED_REL = Path("outputs/c13_final_untouched/corrected_numerical_audit/data")
PRIMARY_COMPLETENESS_REL = Path("outputs/c13_final_untouched/draft32_completeness_audit/data")
PRIMARY_ARRAY_REL = Path("outputs/c13_final_untouched/source_data")
PRIMARY_CALIBRATION_REL = Path("outputs/c13_robust_complete_graph_recalibration_audit/data")
FISHER_ROBUSTNESS_REL = Path("outputs/c13_final_untouched/intervention_fisher_robustness")
LOCATION_PROFILES_REL = Path("outputs/c13_final_untouched/factor_and_complex_location_profiles")
INTER_ELC_REL = Path("outputs/c13_inter_elc_final_untouched/data")
INTER_ELC_CALIBRATION_REL = Path("outputs/c13_inter_elc_calibration/data")
PRIMARY_SEEDS = (254989981, 265358979, 276563425, 287271829)
INTER_ELC_SEEDS = (298463771, 309016999, 320377241, 331662479)


def _read_csv(root: Path, rel: str) -> pd.DataFrame:
    return pd.read_csv(root / rel)


def _fmt(value: float, digits: int = 3) -> str:
    return _sig(float(value), digits)


def _sig(value: float, sig: int = 3) -> str:
    value = float(value)
    if value == 0:
        return "0." + "0" * (sig - 1)
    abs_value = abs(value)
    if abs_value < 1e-3:
        exponent = int(np.floor(np.log10(abs_value)))
        mantissa = value / (10 ** exponent)
        return rf"\({mantissa:.{sig - 1}f}\times10^{{{exponent}}}\)"
    decimals = max(sig - 1 - int(np.floor(np.log10(abs_value))), 0)
    return f"{value:.{decimals}f}"


def _bits(value: float) -> str:
    return _sig(float(value), 3)


def _pct(value: float) -> str:
    return f"{_sig(float(value), 3)}\\%"


def _points(value: float) -> str:
    return f"{float(value):.1f}"


def _prob(value: float) -> str:
    return f"{float(value):.3f}"


def _theta(value: float) -> str:
    value = float(value)
    if value == 0:
        return "0"
    return f"{value:.1f}"


def _pvalue(value: float) -> str:
    value = float(value)
    if np.isclose(value, 0.001996, rtol=0.0, atol=5e-7):
        return "0.002"
    return _sig(value, 3)


def _fmt_cell(value: float) -> str:
    """Three-significant-figure cell labels for heatmaps."""
    value = float(value)
    if value == 0:
        return "0.000"
    if abs(value) < 1e-2:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10 ** exponent)
        return f"{mantissa:.2f}e{exponent}"
    return _sig(value, 3)


def _intervention_cell(value: float) -> str:
    value = float(value)
    if abs(value) < 5e-12:
        return "0.000"
    return _sig(value, 3)


def _intervention_delta_summary(
    surface: pd.DataFrame,
    context_probs: pd.DataFrame,
    *,
    positive: tuple[float, float] = (1.5, -1.5),
    negative: tuple[float, float] = (-1.5, 1.5),
    n_boot: int = 500,
    rng_seed: int = 9173,
) -> pd.DataFrame:
    """Compute context-cluster bootstrap intervals for intervention contrasts."""
    phenotype_col = "p_nu_early_maturation_high_growth"
    p0 = float(surface[(surface["theta_reg"] == 0.0) & (surface["theta_stress"] == 0.0)][phenotype_col].iloc[0])
    key_cols = ["seed", "lineage_id", "tau", "context_id"]
    grid = context_probs[context_probs["on_approved_7x7_response_grid"]].copy()
    wide = grid.pivot_table(index=key_cols, columns=["theta_reg", "theta_stress"], values=phenotype_col)
    wide_df = wide.reset_index()
    cluster = wide_df[["seed", "lineage_id"]].astype(str).agg("::".join, axis=1).to_numpy()
    _, inv = np.unique(cluster, return_inverse=True)
    n_clusters = int(inv.max()) + 1
    rng = np.random.default_rng(rng_seed)

    def one(label: str, theta: tuple[float, float]) -> dict[str, object]:
        p_theta = float(surface[(surface["theta_reg"] == theta[0]) & (surface["theta_stress"] == theta[1])][phenotype_col].iloc[0])
        if theta == (0.0, 0.0):
            return {
                "contrast": label,
                "theta_reg": theta[0],
                "theta_stress": theta[1],
                "p0_nu": p0,
                "p_theta_nu": p_theta,
                "delta_p_nu": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
            }
        delta = wide_df[theta].to_numpy(dtype=float) - wide_df[(0.0, 0.0)].to_numpy(dtype=float)
        sums = np.bincount(inv, weights=delta, minlength=n_clusters)
        counts = np.bincount(inv, minlength=n_clusters)
        draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
        boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        return {
            "contrast": label,
            "theta_reg": theta[0],
            "theta_stress": theta[1],
            "p0_nu": p0,
            "p_theta_nu": p_theta,
            "delta_p_nu": float(delta.mean()),
            "ci_lower": float(np.quantile(boot, 0.025)),
            "ci_upper": float(np.quantile(boot, 0.975)),
        }

    return pd.DataFrame(
        [
            one("reference", (0.0, 0.0)),
            one("selected positive direction", positive),
            one("selected negative direction", negative),
        ]
    )


def _draw_annotated_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    *,
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    fontsize: float = 7.0,
) -> ScalarMappable:
    """Draw a small matrix as vector rectangles to avoid PDF raster artifacts."""
    values = np.asarray(matrix, dtype=float)
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = float(values[i, j])
            rgba = cm(norm(val))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt_color = "white" if luminance < 0.50 else "#111111"
            outline = "#111111" if txt_color == "white" else "white"
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor=rgba, edgecolor="white", linewidth=0.6))
            ax.text(
                j,
                i,
                _fmt(val, 3 if values.shape[0] <= 2 else 2),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=txt_color,
                path_effects=[pe.withStroke(linewidth=0.9, foreground=outline)],
            )
    ax.set_xlim(-0.5, values.shape[1] - 0.5)
    ax.set_ylim(values.shape[0] - 0.5, -0.5)
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels(ylabels)
    ax.set_aspect("equal")
    return ScalarMappable(norm=norm, cmap=cm)


def _tex_escape(value: object) -> str:
    text = str(value)
    if "$" in text or "\\" in text:
        return text
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def _plain_label(value: object) -> str:
    return str(value).replace("_", " ")


def _level_label(value: object) -> str:
    labels = {
        "development": "development",
        "microbiome": "microbiome",
        "life_history": "life history",
        "epigenetic": "epigenetic",
        "ecological": "ecological",
        "background": "background",
    }
    return labels.get(str(value), _plain_label(value))


def _component_label(value: object) -> str:
    text = str(value)
    labels = {
        "gene_1_regulatory_initiation": "regulatory initiation",
        "gene_2_somatic_growth": "somatic growth",
        "gene_3_stress_response": "stress response",
        "gene_4_maturation_timing": "maturation timing",
        "gene_5_allocation_signal": "allocation signal",
        "guild_1_growth_support": "growth-support guild",
        "guild_2_fiber_fermenter": "fiber-fermenting guild",
        "guild_3_stress_tolerant": "stress-tolerant guild",
        "guild_4_opportunist": "opportunist guild",
        "guild_5_cross_feeder": "cross-feeder guild",
        "maturation_progress": "maturation progress",
        "growth_capacity": "growth capacity",
        "reproductive_allocation": "reproductive allocation",
        "z_regulatory_mark": "regulatory mark variable",
        "z_stress_memory_mark": "stress-memory mark variable",
        "soil_organic_matter": "soil organic matter",
        "food_resource_enrichment": "food-resource enrichment",
        "microclimate_buffering": "microclimate buffering",
        "rainfall_anomaly": "rainfall anomaly",
        "temperature_anomaly": "temperature anomaly",
    }
    return labels.get(text, _plain_label(text))


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "figure.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _savefig(fig: plt.Figure, fig_dir: Path, name: str) -> dict[str, str]:
    pdf = fig_dir / f"{name}.pdf"
    png = fig_dir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=FIG_DPI)
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def _copy_sanitized(src: Path, dst: Path, project_root: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {".csv", ".json", ".txt", ".md", ".tex", ".bib", ".py", ".log"}:
        text = src.read_text(errors="ignore")
        text = text.replace(str(project_root) + os.sep, "./")
        text = text.replace(str(project_root), ".")
        text = text.replace(str(project_root.parent) + os.sep, "../")
        user_path_pattern = re.compile(r"/" + r"Users/[^\s,'\";)]+(?:/[^\s,'\";)]+)*")
        text = user_path_pattern.sub(".", text)
        text = text.replace("Draft 32", "the accompanying article")
        text = text.replace("corrected estimator", "final estimator")
        text = text.replace("corrected estimators", "final estimators")
        text = text.replace("selected pipeline", "selected estimator")
        text = text.replace("null" + "-excess", "information above surrogate mean")
        text = text.replace("binary nu", r"binary \(\nu\)")
        text = text.replace("focal state nu", r"focal state \(\nu\)")
        text = text.replace("focal log-ratio among nu", r"focal log-ratio among \(\nu\)")
        text = text.replace("p(nu)", r"\(p(\nu)\)")
        text = text.replace("V=nu", r"V=\nu")
        text = text.replace("nu_early_maturation_high_growth", r"\(\nu\): early maturation and high growth")
        text = text.replace("early_maturation_low_growth", "early maturation and low growth")
        text = text.replace("non_early_maturation_high_growth", "non-early maturation and high growth")
        text = text.replace("non_early_maturation_low_growth", "non-early maturation and low growth")
        text = text.replace("Delta p_nu^+", r"\(\Delta p_\nu^+\)")
        dst.write_text(text)
    else:
        shutil.copy2(src, dst)


def _copy_tree_sanitized(src: Path, dst: Path, project_root: Path, *, include_suffixes: set[str] | None = None) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if path.is_dir():
            if path.name in {"__pycache__", ".pytest_cache"}:
                continue
            continue
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if include_suffixes is not None and path.suffix.lower() not in include_suffixes:
            continue
        rel = path.relative_to(src)
        _copy_sanitized(path, dst / rel, project_root)


def _prune_obsolete_inter_elc_artifacts(root: Path) -> None:
    """Remove superseded paired-lineage artifacts from generated deliverables."""
    if not root.exists():
        return
    obsolete_tokens = ("multiple_parent", "multiparent", "deranged")
    for path in sorted(root.rglob("*"), reverse=True):
        name = path.name.lower()
        if any(token in name for token in obsolete_tokens):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)


def _prune_rowlevel_score_tables(root: Path) -> None:
    """Keep delivered tables focused on summaries, figure sources, and frozen arrays."""
    if not root.exists():
        return
    for path in root.rglob("*crossfit_scores.csv"):
        if path.is_file():
            path.unlink()


def _prune_copy_artifacts(root: Path) -> None:
    """Remove stale finder-style copy artifacts from generated deliverables."""
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if " copy" in path.name.lower():
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)


def _zip_package(package_dir: Path, archive: Path) -> None:
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, filenames in os.walk(package_dir):
            for filename in filenames:
                path = Path(root) / filename
                if path.is_file():
                    zf.write(path, path.relative_to(package_dir.parent))


def _remove_empty_dirs(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def _figure_source(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)


def _load_primary_arrays(project_root: Path, seed: int = PRIMARY_SEEDS[0]) -> dict[str, np.ndarray]:
    path = project_root / PRIMARY_ARRAY_REL / f"corrected_temporal_architecture_seed_{seed}.npz"
    loaded = np.load(path)
    arrays = {key: loaded[key] for key in loaded.files}
    for level in ["development", "microbiome", "life_history", "epigenetic", "ecological", "background"]:
        arrays[level] = arrays[f"segment_{level}"]
    return arrays


def _plot_multilevel_dynamics(project_root: Path, fig_dir: Path, table_dir: Path) -> str:
    seed = PRIMARY_SEEDS[0]
    arrays = _load_primary_arrays(project_root, seed)
    levels = ["development", "microbiome", "life_history", "epigenetic", "ecological", "background"]
    ylabels = {
        "development": "state value",
        "microbiome": "log(1 + abundance)",
        "life_history": "state value",
        "epigenetic": "state value",
        "ecological": "state value",
        "background": "state value",
    }
    titles = {
        "development": "Development",
        "microbiome": "Microbiome",
        "life_history": "Life history",
        "epigenetic": "Epigenetic variables",
        "ecological": "Ecological state",
        "background": "Background",
    }
    rows = []
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 7.5), constrained_layout=True)
    lineage = 0
    generation = 9
    for ax, level in zip(axes.flat, levels):
        arr = arrays[f"full_{level}"][lineage, generation]
        timestamps = arrays[f"timestamps_{level}"]
        segment_indices = arrays[f"segment_indices_{level}"].astype(int)
        reproductive_index = int(arrays[f"reproductive_index_{level}"])
        m_l = arr.shape[0]
        dims = arr.shape[1]
        x = timestamps
        y = arr.copy()
        if level == "microbiome":
            y = np.log1p(y)
        for j in range(dims):
            raw_name = COMPONENTS[level][j]
            name = _component_label(raw_name)
            ax.plot(x, y[:, j], lw=1.2, label=name)
            for k, val in enumerate(y[:, j]):
                rows.append(
                    {
                        "figure": "figure01_multilevel_dynamics",
                        "seed": seed,
                        "illustrative_lineage_d": 1,
                        "level": level,
                        "generation": generation,
                        "t_l_index": int(k),
                        "u": float(x[k]),
                        "in_analysis_segment": bool(k in set(segment_indices.tolist())),
                        "is_reproductive_state": bool(k == reproductive_index),
                        "component": name,
                        "value": float(val),
                    }
                )
        ax.set_title(titles[level])
        ax.axvspan(x[segment_indices[0]], x[segment_indices[-1]], color="#F2C94C", alpha=0.14)
        ax.axvline(0.75, color="#222222", ls="--", lw=0.9)
        ax.set_xlabel(r"within-generation time $t_l$ in generation 9")
        ax.set_ylabel(ylabels[level])
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _figure_source(pd.DataFrame(rows), table_dir, "figure01_multilevel_dynamics")
    _savefig(fig, fig_dir, "figure01_multilevel_dynamics")
    return "figure01_multilevel_dynamics"


def _plot_phenotype_reconstruction(project_root: Path, fig_dir: Path, table_dir: Path) -> str:
    seed = PRIMARY_SEEDS[0]
    arrays = _load_primary_arrays(project_root, seed)
    # This trajectory realizes the focal phenotype in generation 9.
    illustrative_lineage_index = 0
    life = arrays["full_life_history"][illustrative_lineage_index, 9]
    t = arrays["timestamps_life_history"]
    mat = life[:, 0]
    grow = life[:, 1]
    theta_m = 0.60
    theta_g = 0.65
    u_early = 0.60
    crossing = np.where(mat >= theta_m)[0]
    t_mat = int(crossing[0]) if len(crossing) else None
    u_mat = float(t[t_mat]) if t_mat is not None else None
    final_growth = float(grow[-1])
    is_early = u_mat is not None and u_mat < u_early
    is_high_growth = final_growth >= theta_g
    rows = []
    for idx, u in enumerate(t):
        rows.append(
            {
                "figure": "figure03_phenotype_reconstruction",
                "seed": seed,
                "illustrative_lineage_d": illustrative_lineage_index + 1,
                "generation_tau_plus_1": 9,
                "t_l_index": int(idx),
                "u": float(u),
                "maturation_progress": float(mat[idx]),
                "growth_capacity": float(grow[idx]),
                "theta_M": theta_m,
                "u_early": u_early,
                "theta_G": theta_g,
                "maturation_crossing_time": -1 if t_mat is None else t_mat,
                "early_maturation": bool(is_early),
                "high_growth": bool(is_high_growth),
                "phenotype_realization": "nu_(1,1)" if is_early and is_high_growth else f"({int(is_early)},{int(is_high_growth)})",
            }
        )
    _figure_source(pd.DataFrame(rows), table_dir, "figure03_phenotype_reconstruction")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True, sharex=True)
    ax = axes[0]
    ax.plot(t, mat, marker="o", color=PALETTE["life_history"], lw=1.8, label="maturation progress")
    ax.axhline(theta_m, color="#333333", ls="--", lw=1.0, label=r"$\theta_M=0.60$")
    ax.axvspan(0.0, u_early, color="#F2C94C", alpha=0.14, label=r"early window $u<0.60$")
    if t_mat is not None:
        ax.scatter([t[t_mat]], [mat[t_mat]], s=44, color="#111111", zorder=5)
    ax.set_title("Maturation threshold crossing")
    ax.set_ylabel("maturation progress")
    ax.set_xlabel(r"normalized within-generation time $u$")
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0.15, 0.80)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.28), frameon=False, ncol=1)

    ax = axes[1]
    ax.plot(t, grow, marker="o", color="#0072B2", lw=1.8, label="growth capacity")
    ax.axhline(theta_g, color="#333333", ls="--", lw=1.0, label=r"$\theta_G=0.65$")
    ax.scatter([t[-1]], [final_growth], s=44, color="#111111", zorder=5, label=r"growth at $u=1$")
    ax.set_title("Final growth criterion")
    ax.set_ylabel("growth capacity")
    ax.set_xlabel(r"normalized within-generation time $u$")
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0.43, 0.72)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.28), frameon=False, ncol=1)
    fig.suptitle(r"Reconstruction of $V_{d,\tau+1}=\nu$ from the generated life-history time series", y=1.03)
    _savefig(fig, fig_dir, "figure03_phenotype_reconstruction")
    return "figure03_phenotype_reconstruction"


def _plot_architecture(project_root: Path, fig_dir: Path, table_dir: Path) -> str:
    edges = pd.read_csv(project_root / PRIMARY_CORRECTED_REL / "corrected_final_nonzero_cross_level_edges.csv")
    _figure_source(edges, table_dir, "figure02_architecture_edges")
    pos = {
        "development": (0.25, 0.73),
        "microbiome": (0.25, 0.49),
        "life_history": (0.25, 0.25),
        "epigenetic": (0.63, 0.63),
        "ecological": (0.63, 0.35),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    boundary = FancyBboxPatch(
        (0.04, 0.08),
        0.72,
        0.84,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#222222",
        facecolor="#FFFFFF",
        alpha=0.96,
        zorder=0,
    )
    ax.add_patch(boundary)
    ax.text(0.065, 0.885, "Extended Life Cycle", fontsize=10, weight="bold", zorder=2)

    coupling_box = FancyBboxPatch(
        (0.415, 0.455),
        0.17,
        0.09,
        boxstyle="round,pad=0.016,rounding_size=0.018",
        linewidth=0.9,
        edgecolor="#777777",
        facecolor="#F7F7F7",
        zorder=2,
    )
    ax.add_patch(coupling_box)
    ax.text(0.500, 0.500, "within-ELC\ncouplings $C^{[l]}$", ha="center", va="center", fontsize=7.3, zorder=3)

    for node, (x, y) in pos.items():
        color = PALETTE[node]
        box = FancyBboxPatch(
            (x - 0.14, y - 0.052),
            0.28,
            0.104,
            boxstyle="round,pad=0.014,rounding_size=0.014",
            linewidth=1.1,
            edgecolor=color,
            facecolor=matplotlib.colors.to_rgba(color, 0.08),
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y, node.replace("_", "-"), ha="center", va="center", fontsize=8.7, zorder=4)

    # Component-level edges are recorded in the source table.
    for node, (x, y) in pos.items():
        color = PALETTE[node]
        arrow = FancyArrowPatch(
            (0.500, 0.500),
            (x, y),
            arrowstyle="-",
            mutation_scale=6,
            lw=0.65,
            color="#8A8A8A",
            alpha=0.35,
            zorder=1,
        )
        ax.add_patch(arrow)

    bg_box = FancyBboxPatch(
        (0.82, 0.425),
        0.14,
        0.13,
        boxstyle="round,pad=0.014,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=PALETTE["background"],
        facecolor="#F2F2F2",
        zorder=3,
    )
    ax.add_patch(bg_box)
    ax.text(0.89, 0.490, "background", ha="center", va="center", fontsize=8.4, zorder=4)
    ax.text(0.82, 0.59, "outside ELC", fontsize=8.0, weight="bold", color="#444444", zorder=2)
    bg_arrow = FancyArrowPatch(
        (0.82, 0.49),
        (0.762, 0.49),
        arrowstyle="-|>",
        mutation_scale=9,
        lw=1.0,
        color=PALETTE["background"],
        alpha=0.75,
        zorder=2,
    )
    ax.add_patch(bg_arrow)
    source_box = FancyBboxPatch(
        (0.80, 0.78),
        0.16,
        0.11,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=PALETTE["epigenetic"],
        facecolor=matplotlib.colors.to_rgba(PALETTE["epigenetic"], 0.10),
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(source_box)
    ax.text(0.88, 0.835, "additional source\nlineage $d'$", ha="center", va="center", fontsize=7.2, zorder=3)
    b_arrow = FancyArrowPatch(
        (0.80, 0.815),
        (pos["epigenetic"][0] + 0.14, pos["epigenetic"][1] + 0.02),
        arrowstyle="-|>",
        mutation_scale=9,
        lw=1.0,
        color=PALETTE["epigenetic"],
        linestyle="--",
        alpha=0.85,
        connectionstyle="arc3,rad=0.05",
        zorder=1,
    )
    ax.add_patch(b_arrow)
    ax.text(0.735, 0.755, r"$B^{[\mathrm{epi}]}_{d,d'}$", fontsize=7.0, color=PALETTE["epigenetic"])
    _savefig(fig, fig_dir, "figure02_coupled_elc_architecture")
    return "figure02_coupled_elc_architecture"


def _plot_graph_reconstruction(project_root: Path, fig_dir: Path, table_dir: Path) -> list[str]:
    graph_dir = project_root / "outputs" / "c13_final_untouched" / "graph_reconstruction"
    matrix_path = graph_dir / "graph_reconstruction_information_matrix.csv"
    summary_path = graph_dir / "graph_reconstruction_recovery_summary.csv"
    config = GraphReconstructionConfig(seeds=PRIMARY_SEEDS)
    if matrix_path.exists() and summary_path.exists():
        matrix = pd.read_csv(matrix_path)
        matrix, summary = reclassify_graph_dependencies(matrix, config)
        matrix.to_csv(matrix_path, index=False)
        summary.to_csv(summary_path, index=False)
        known_dependency_table().to_csv(graph_dir / "graph_reconstruction_known_dependencies.csv", index=False)
    else:
        graph = run_final_conditional_dependence_reconstruction(project_root, graph_dir, config)
        matrix = graph["matrix"]
    summary_values = pd.read_csv(summary_path)
    detection_alpha = float(summary_values.iloc[0]["alpha"])
    _figure_source(matrix, table_dir, "figure03_coupling_networks")
    _figure_source(matrix, table_dir, "figure04_coupling_information_matrix")
    sources = ["development", "microbiome", "life_history", "epigenetic", "ecological", "background"]
    targets = sources
    known = matrix.pivot(index="target_level", columns="source_level", values="known_dependency").loc[targets, sources]
    recovered = matrix.pivot(index="target_level", columns="source_level", values="recovered_dependency").loc[targets, sources]
    excess = matrix.pivot(index="target_level", columns="source_level", values="null_excess_bits").loc[targets, sources]

    def draw_network(ax: plt.Axes, title: str, detected_panel: bool = False) -> None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, pad=8)
        y_pos = {level: 0.88 - i * 0.145 for i, level in enumerate(sources)}
        left_x, right_x = 0.20, 0.78
        box_w, box_h = 0.30, 0.075
        ax.text(left_x, 0.995, "sources at $\\tau$", ha="center", va="top", fontsize=7.5, weight="bold")
        ax.text(right_x, 0.995, "targets at $\\tau+1$", ha="center", va="top", fontsize=7.5, weight="bold")
        for level in sources:
            y = y_pos[level]
            color = PALETTE.get(level, "#777777")
            for x, prefix in [(left_x, "current"), (right_x, "future")]:
                label = f"{prefix}\n{_level_label(level)}"
                patch = FancyBboxPatch(
                    (x - box_w / 2, y - box_h / 2),
                    box_w,
                    box_h,
                    boxstyle="round,pad=0.025,rounding_size=0.018",
                    fc=matplotlib.colors.to_rgba(color, 0.12),
                    ec=color,
                    lw=0.9,
                    zorder=3,
                )
                ax.add_patch(patch)
                ax.text(x, y, label, ha="center", va="center", fontsize=6.2, zorder=4)
        max_excess = float(np.nanmax(excess.to_numpy(dtype=float))) if detected_panel else 1.0
        max_excess = max(max_excess, 1e-12)
        for src in sources:
            for tgt in targets:
                row = matrix[(matrix["source_level"] == src) & (matrix["target_level"] == tgt)].iloc[0]
                if detected_panel:
                    include = int(row["recovered_dependency"]) == 1
                    additional = str(row["edge_status"]) == "additional_inferred"
                else:
                    include = int(row["known_dependency"]) == 1
                    additional = False
                if not include:
                    continue
                y0, y1 = y_pos[src], y_pos[tgt]
                color = PALETTE.get(src, "#555555")
                weight = float(excess.loc[tgt, src]) / max_excess if detected_panel else 0.55
                lw = 0.9 + 2.2 * max(weight, 0.03)
                alpha = 0.42 + 0.45 * max(weight, 0.05)
                arrow = FancyArrowPatch(
                    (left_x + box_w / 2, y0),
                    (right_x - box_w / 2, y1),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    lw=lw,
                    color=color,
                    alpha=alpha,
                    linestyle="--" if additional else "-",
                    connectionstyle="arc3,rad=0.05",
                    zorder=1,
                )
                ax.add_patch(arrow)
        if detected_panel:
            legend_handles = [
                Line2D([0], [0], color="#555555", lw=1.8, ls="-", label="specified dependency recovered"),
                Line2D([0], [0], color="#555555", lw=1.8, ls="--", label="additional conditional association"),
            ]
            ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=6.4)

    fig_network = plt.figure(figsize=(7.2, 7.4), constrained_layout=True)
    gs = fig_network.add_gridspec(2, 1, height_ratios=[1.0, 1.0])
    ax_known = fig_network.add_subplot(gs[0, 0])
    ax_rec = fig_network.add_subplot(gs[1, 0])
    draw_network(ax_known, "(a) dependencies specified in the equations", detected_panel=False)
    draw_network(ax_rec, "(b) conditional associations detected by the estimator", detected_panel=True)
    _savefig(fig_network, fig_dir, "figure03_coupling_networks")

    fig_matrix, ax_mat = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    excess_values = excess.to_numpy(dtype=float)
    vmax = max(float(np.nanmax(excess_values)), 1e-12)
    matrix_cmap = plt.get_cmap("viridis").copy()
    matrix_cmap.set_bad("#E3E3E3")
    pvals = matrix.pivot(
        index="target_level",
        columns="source_level",
        values="generation_preserving_surrogate_p_value",
    ).loc[targets, sources].to_numpy(dtype=float)
    significant = pvals <= detection_alpha
    display_values = np.ma.masked_where(~significant, excess_values)
    im = ax_mat.imshow(display_values, aspect="equal", cmap=matrix_cmap, norm=Normalize(vmin=0.0, vmax=vmax))
    ax_mat.set_title("Information beyond randomized baseline", pad=8)
    ax_mat.set_xlabel("sources at $\\tau$")
    ax_mat.set_ylabel("targets at $\\tau+1$")
    ax_mat.set_xticks(range(len(sources)))
    ax_mat.set_xticklabels([_level_label(s) for s in sources], rotation=35, ha="right", fontsize=9.0)
    ax_mat.set_yticks(range(len(targets)))
    ax_mat.set_yticklabels([_level_label(t) for t in targets], fontsize=9.0)
    for i, tgt in enumerate(targets):
        for j, src in enumerate(sources):
            val = float(excess.loc[tgt, src])
            is_sig = bool(significant[i, j])
            rgba = im.cmap(im.norm(max(val, 0.0))) if is_sig else matplotlib.colors.to_rgba("#E3E3E3")
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt_color = "white" if luminance < 0.40 else "#111111"
            label = _fmt_cell(val)
            outline = "black" if txt_color == "white" else "#F8F8F8"
            ax_mat.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8.9,
                color=txt_color,
                path_effects=[pe.withStroke(linewidth=0.8, foreground=outline)],
            )
            status = str(matrix[(matrix["source_level"] == src) & (matrix["target_level"] == tgt)].iloc[0]["edge_status"])
            if status == "expected_recovered":
                rect = Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec="#111111", lw=1.2)
                ax_mat.add_patch(rect)
            elif status == "additional_inferred":
                rect = Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec="#111111", lw=1.2, ls="--")
                ax_mat.add_patch(rect)
            elif status == "expected_absent":
                rect = Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec="#888888", lw=0.8, ls=":")
                ax_mat.add_patch(rect)
    fig_matrix.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04, label="bits beyond randomized baseline")
    ax_mat.legend(
        handles=[
            Patch(facecolor="none", edgecolor="#111111", lw=1.2, label="specified dependency recovered"),
            Patch(facecolor="none", edgecolor="#111111", lw=1.2, linestyle="--", label="additional conditional association"),
            Patch(
                facecolor="#E3E3E3",
                edgecolor="#888888",
                lw=0.8,
                linestyle=":",
                label=rf"specified-absent; not detected ($p>{detection_alpha:.2f}$)",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=False,
        ncol=1,
        fontsize=7.2,
    )
    _savefig(fig_matrix, fig_dir, "figure04_coupling_information_matrix")
    return ["figure03_coupling_networks", "figure04_coupling_information_matrix"]


def _plot_predictive_contribution(project_root: Path, fig_dir: Path, table_dir: Path) -> str:
    df = pd.read_csv(project_root / PRIMARY_CORRECTED_REL / "coherent_continuous_transfer_entropy_summary.csv")
    keep = df[df["analysis"].isin([
        "epigenetic_predictive_contribution",
        "background_predictive_contribution",
        "epigenetic_predictive_closure",
        "background_closure_control",
    ])].copy()
    label_map = {
        "epigenetic_predictive_contribution": "epigenetic contribution",
        "background_predictive_contribution": "background contribution",
        "epigenetic_predictive_closure": "epigenetic dependence",
        "background_closure_control": "background dependence",
    }
    keep["label"] = keep["analysis"].map(label_map)
    _figure_source(keep, table_dir, "figure04_predictive_contribution_closure")
    x = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=(6.9, 3.1), constrained_layout=True)
    raw = keep["raw_estimate_bits"].to_numpy(dtype=float)
    lower = keep["ci_lower_bits"].to_numpy(dtype=float)
    upper = keep["ci_upper_bits"].to_numpy(dtype=float)
    pvals = keep["generation_preserving_surrogate_p_value"].to_numpy(dtype=float)
    significant = pvals <= 0.05
    for i in range(len(keep)):
        color = "#0072B2" if significant[i] else "#777777"
        ax.errorbar(
            x[i] - 0.08,
            raw[i],
            yerr=[[raw[i] - lower[i]], [upper[i] - raw[i]]],
            fmt="o",
            mfc=color if significant[i] else "white",
            mec=color,
            color=color,
            capsize=3,
            zorder=4,
        )
    ax.scatter(x + 0.08, keep["surrogate_null_mean_bits"].to_numpy(dtype=float), color=PALETTE["null"], marker="s", label="permutation baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(keep["label"], rotation=25, ha="right")
    ax.set_ylabel("conditional mutual information (bits)")
    ax.set_title("Predictive contribution and dependence on the ELC relative to permutation baselines")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#0072B2", markerfacecolor="#0072B2", lw=0, label="distinguishable from baseline"),
            Line2D([0], [0], marker="o", color="#777777", markerfacecolor="white", lw=0, label="not distinguishable from baseline"),
            Line2D([0], [0], marker="s", color=PALETTE["null"], markerfacecolor=PALETTE["null"], lw=0, label="permutation baseline"),
        ],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=7.5,
    )
    _savefig(fig, fig_dir, "figure04_predictive_contribution_closure")
    return "figure04_predictive_contribution_closure"


def _plot_stability(project_root: Path, fig_dir: Path, table_dir: Path) -> str:
    stab = pd.read_csv(project_root / PRIMARY_CORRECTED_REL / "corrected_intergenerational_stability_summary.csv")
    _figure_source(stab, table_dir, "figure07_intergenerational_stability")
    fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
    ax.errorbar(
        stab["rho"].to_numpy(dtype=float),
        stab["raw_estimate_bits"].to_numpy(dtype=float),
        yerr=[(stab["raw_estimate_bits"] - stab["ci_lower_bits"]).to_numpy(dtype=float), (stab["ci_upper_bits"] - stab["raw_estimate_bits"]).to_numpy(dtype=float)],
        marker="o",
        lw=1.2,
        color=PALETTE["epigenetic"],
        capsize=3,
    )
    ax.set_xlabel("horizon $\\rho$")
    ax.set_xticks(stab["rho"].to_numpy(dtype=int))
    ax.set_ylabel("bits")
    ax.set_title("Epigenetic predictive contribution across intergenerational horizons")
    ax.grid(alpha=0.18)
    _savefig(fig, fig_dir, "figure07_intergenerational_stability")
    return "figure07_intergenerational_stability"


def _plot_location(project_root: Path, fig_dir: Path, table_dir: Path) -> list[str]:
    loc = pd.read_csv(project_root / LOCATION_PROFILES_REL / "factor_and_complex_location_summary.csv")
    eligible = (
        ((loc["source_profile"] == "epigenetic_factor") & (loc["target_level"] != "epigenetic"))
        | (
            (loc["source_profile"] == "joint_epigenetic_ecological")
            & (~loc["target_level"].isin(["epigenetic", "ecological"]))
        )
    )
    location = loc[eligible].copy()
    auxiliary = loc[~eligible].copy()
    auxiliary.insert(0, "analysis_role", "auxiliary source-persistence or self-prediction diagnostic")
    location_figure_source = location.copy()
    location_figure_source["time_index_j"] = location_figure_source["target_t_l"]
    location_figure_source["u_l_j"] = location_figure_source["target_u"]
    _figure_source(location_figure_source, table_dir, "figure08_level_time_location")
    location.to_csv(table_dir.parent / "factor_and_complex_location_summary.csv", index=False)
    auxiliary.to_csv(table_dir.parent / "source_level_persistence_profiles.csv", index=False)
    source_specs = [
        (
            "epigenetic_factor",
            ["development", "microbiome", "life_history", "ecological"],
            "Epigenetic-factor predictive contribution across within-generation time",
            "figure08_level_time_location",
            (5.325, 8.0),
            (4, 1),
        ),
        (
            "joint_epigenetic_ecological",
            ["development", "microbiome", "life_history"],
            "Joint epigenetic--ecological predictive contribution across within-generation time",
            "figure08_joint_epigenetic_ecological_location",
            (5.62, 6.3),
            (3, 1),
        ),
    ]

    row_limits: dict[str, tuple[float, float]] = {}
    for level in ["development", "microbiome", "life_history", "ecological"]:
        row_values = location[location["target_level"] == level]
        if level == "ecological":
            row_values = row_values[row_values["source_profile"] == "epigenetic_factor"]
        low = float((row_values["ci_lower_bits"] - row_values["randomized_source_mean_bits"]).min())
        high = float((row_values["ci_upper_bits"] - row_values["randomized_source_mean_bits"]).max())
        span = max(high - low, 1e-6)
        row_limits[level] = (min(0.0, low) - 0.06 * span, max(0.0, high) + 0.06 * span)

    output_names: list[str] = []
    for source_name, target_levels, title, filename, figsize, grid_shape in source_specs:
        nrows, ncols = grid_shape
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, constrained_layout=False)
        axes = np.atleast_1d(axes).ravel()
        for row, level in enumerate(target_levels):
            ax = axes[row]
            d = location[
                (location["target_level"] == level)
                & (location["source_profile"] == source_name)
            ].sort_values("target_t_l")
            x = d["target_u"].to_numpy(dtype=float)
            y = d["information_beyond_randomized_mean_bits"].to_numpy(dtype=float)
            lower = d["ci_lower_bits"].to_numpy(dtype=float) - d["randomized_source_mean_bits"].to_numpy(dtype=float)
            upper = d["ci_upper_bits"].to_numpy(dtype=float) - d["randomized_source_mean_bits"].to_numpy(dtype=float)
            color = PALETTE.get(level, "#333333")
            ax.fill_between(x, lower, upper, color=color, alpha=0.17, linewidth=0)
            ax.plot(x, y, lw=1.1, color=color)
            ax.axhline(0.0, color="#777777", lw=0.7, ls="--", zorder=0)
            sig = d["randomized_source_p_value"].to_numpy(dtype=float) <= 0.05
            if np.any(~sig):
                ax.scatter(x[~sig], y[~sig], marker="o", s=24, facecolor="white", edgecolor="#666666", linewidth=0.9, zorder=6)
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
            ax.grid(alpha=0.14)
            ax.set_ylabel(
                f"{_level_label(level)}\ninformation beyond\nbaseline (bits)",
                fontsize=8,
            )
            ax.set_ylim(*row_limits[level])
            if row >= (nrows - 1) * ncols:
                ax.set_xlabel(r"normalized time $u_{l,j}=j/(T_l-1)$")
        for ax in axes[len(target_levels):]:
            ax.axis("off")
        fig.legend(
            handles=[
                Line2D([0], [0], color="#555555", lw=1.2, label=r"time-specific estimate; randomized-source $p\leq0.05$ unless marked"),
                Line2D([0], [0], marker="o", color="#666666", markerfacecolor="white", lw=0, label=r"not distinguishable from baseline ($p>0.05$)"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=1,
            frameon=False,
            fontsize=7.5,
        )
        fig.subplots_adjust(left=0.19, right=0.985, top=0.94, bottom=0.13, hspace=0.24)
        fig.suptitle(title, y=0.985, fontsize=11)
        _savefig(fig, fig_dir, filename)
        output_names.append(filename)
    return output_names


def _plot_pid(project_root: Path, fig_dir: Path, table_dir: Path) -> str:
    pid = pd.read_csv(project_root / PRIMARY_CORRECTED_REL / "corrected_epigenetic_ecological_delta_g_pid_summary.csv").iloc[0]
    ci = pd.read_csv(project_root / PRIMARY_CORRECTED_REL / "corrected_epigenetic_ecological_delta_g_pid_centered_intervals.csv")
    rows = pd.DataFrame(
        [
            {"atom": "redundancy", "bits": pid["redundancy_bits"]},
            {"atom": "unique epigenetic", "bits": pid["unique_source_1_bits"]},
            {"atom": "unique ecological", "bits": pid["unique_source_2_bits"]},
            {"atom": "synergy", "bits": pid["synergy_bits"]},
        ]
    )
    _figure_source(rows, table_dir, "figure06_pid_atoms")
    fig, ax = plt.subplots(figsize=(5.3, 3.0), constrained_layout=True)
    colors = ["#999999", PALETTE["epigenetic"], PALETTE["ecological"], "#56B4E9"]
    ax.bar(rows["atom"], rows["bits"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_ylabel("PID information (bits)")
    ax.set_title("Partial information decomposition of epigenetic and ecological contributions")
    ax.tick_params(axis="x", rotation=20)
    ax.text(0.98, 0.94, f"total = {_fmt(pid['matched_joint_information_bits'])} bits", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    _savefig(fig, fig_dir, "figure06_epigenetic_ecological_pid")
    return "figure06_epigenetic_ecological_pid"


def _plot_variant_intervention_fisher(project_root: Path, fig_dir: Path, table_dir: Path) -> str:
    cat = pd.read_csv(project_root / PRIMARY_CORRECTED_REL / "categorical_phenotype_information_summary.csv")
    surface = pd.read_csv(project_root / PRIMARY_CORRECTED_REL / "intervention_generation_stratified_response_surface_summary.csv")
    contexts = pd.read_csv(project_root / PRIMARY_CORRECTED_REL / "intervention_generation_stratified_probability_by_context.csv")
    delta_summary = _intervention_delta_summary(surface, contexts)
    _figure_source(cat, table_dir, "figure07a_epigenetic_variant_information")
    _figure_source(surface, table_dir, "figure07b_epigenetic_intervention_surface")
    _figure_source(delta_summary, table_dir, "figure07c_epigenetic_intervention_delta_summary")
    p0 = float(delta_summary.loc[delta_summary["contrast"] == "reference", "p0_nu"].iloc[0])
    surface = surface.copy()
    surface["delta_p_nu"] = surface["p_nu_early_maturation_high_growth"] - p0
    delta_pivot = surface.pivot(index="theta_stress", columns="theta_reg", values="delta_p_nu").sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
    xvals = delta_pivot.columns.to_numpy(dtype=float)
    yvals = delta_pivot.index.to_numpy(dtype=float)
    max_abs = max(float(np.nanmax(np.abs(delta_pivot.to_numpy(dtype=float)))), 1e-12)
    dx = float(np.median(np.diff(xvals))) if len(xvals) > 1 else 1.0
    dy = float(np.median(np.diff(yvals))) if len(yvals) > 1 else 1.0
    vals = delta_pivot.to_numpy(dtype=float)
    im = ax.imshow(
        vals,
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
        origin="lower",
        extent=[xvals.min() - dx / 2, xvals.max() + dx / 2, yvals.min() - dy / 2, yvals.max() + dy / 2],
        aspect="equal",
    )
    for yi, yv in enumerate(yvals):
        for xi, xv in enumerate(xvals):
            val = vals[yi, xi]
            rgba = im.cmap(im.norm(val))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "white" if luminance < 0.46 else "#111111"
            outline = "black" if color == "white" else "white"
            ax.text(
                xv,
                yv,
                _intervention_cell(float(val)),
                ha="center",
                va="center",
                fontsize=7.2,
                color=color,
                path_effects=[pe.withStroke(linewidth=1.0, foreground=outline)],
            )
    ax.set_xlabel("$\\theta_{\\mathrm{reg}}$")
    ax.set_ylabel("$\\theta_{\\mathrm{stress}}$")
    ax.set_xticks(xvals)
    ax.set_yticks(yvals)
    ax.set_title(r"Change in $p(V_{d,\tau+1}=\nu)$ under epigenetic interventions", pad=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05, label="$\\Delta p_\\nu(\\theta_x)$")
    _savefig(fig, fig_dir, "figure07_epigenetic_variant_intervention_fisher")
    return "figure07_epigenetic_variant_intervention_fisher"


def _plot_complex_unit(project_root: Path, fig_dir: Path, table_dir: Path) -> str:
    qual = pd.read_csv(project_root / PRIMARY_COMPLETENESS_REL / "complex_unit_qualification_table.csv")
    if "result" in qual.columns or "requirement" in qual.columns:
        qual = qual.copy()
        if "result" in qual.columns:
            qual["result"] = qual["result"].map(_publication_result_text)
        if "requirement" in qual.columns:
            qual["requirement"] = qual["requirement"].map(_publication_result_text)

    def figure_table_text(value: object) -> str:
        text = str(value)
        replacements = {
            r"\(\nu\)": "nu",
            r"\(p(\nu)\)": "p(nu)",
            r"\(\Delta p_\nu^+\)": "Delta p_nu+",
            r"\(\Delta p_\nu^-\)": "Delta p_nu-",
            "Delta p_nu+": "positive change",
            "Delta p_nu-": "negative change",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    _figure_source(qual, table_dir, "figure08d_complex_qualification")
    fig, ax = plt.subplots(figsize=(7.1, 3.4), constrained_layout=True)
    ax.axis("off")
    compact = qual[["requirement", "result", "pass_fail"]].copy()
    compact["requirement"] = compact["requirement"].map(figure_table_text)
    compact["result"] = compact["result"].map(figure_table_text)
    compact["requirement"] = compact["requirement"].map(lambda x: textwrap.fill(str(x), 28))
    compact["result"] = compact["result"].map(lambda x: textwrap.fill(str(x), 54))
    compact["pass_fail"] = compact["pass_fail"].map(lambda x: "pass" if str(x).lower() == "pass" else str(x))
    table = ax.table(
        cellText=compact.values,
        colLabels=["criterion", "result", "status"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.27, 0.58, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.7)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(weight="bold")
        elif col == 2:
            cell.set_facecolor("#E8F5E9")
    ax.set_title("Six-criterion complex-unit qualification", pad=8)
    _savefig(fig, fig_dir, "figure08_complex_unit_qualification")
    return "figure08_complex_unit_qualification"


def _plot_inter_elc_contribution(project_root: Path, fig_dir: Path, table_dir: Path) -> list[str]:
    summary = pd.read_csv(project_root / INTER_ELC_REL / "inter_elc_final_whole_and_subsystem_summary.csv")
    profile = pd.read_csv(project_root / INTER_ELC_REL / "inter_elc_final_target_level_profile_summary.csv")
    _figure_source(summary, table_dir, "figure09a_inter_elc_source_contribution")
    _figure_source(profile, table_dir, "figure09b_inter_elc_target_profiles")
    label_map = {
        "whole_source_elc_from_d_prime": "complete ELC of $d'$",
        "dprime_epigenetic_subsystem": "selected epigenetic segment of $d'$",
        "dprime_microbiome_subsystem": "selected microbiome segment of $d'$",
        "dprime_ecological_subsystem": "selected ecological segment of $d'$",
    }
    summary = summary.copy()
    summary["label"] = summary["analysis"].map(label_map)
    order = [
        "whole_source_elc_from_d_prime",
        "dprime_epigenetic_subsystem",
        "dprime_microbiome_subsystem",
        "dprime_ecological_subsystem",
    ]
    summary = summary.set_index("analysis").loc[order].reset_index()
    y = np.arange(len(summary))

    fig1, ax1 = plt.subplots(figsize=(7.1, 3.4), constrained_layout=True)
    raw = summary["raw_estimate_bits"].to_numpy(dtype=float)
    lower = summary["ci_lower_bits"].to_numpy(dtype=float)
    upper = summary["ci_upper_bits"].to_numpy(dtype=float)
    baseline = summary["surrogate_null_mean_bits"].to_numpy(dtype=float)
    significant = summary["generation_preserving_surrogate_p_value"].to_numpy(dtype=float) <= 0.05
    for i in range(len(summary)):
        color = "#0072B2" if significant[i] else "#777777"
        ax1.errorbar(
            raw[i],
            y[i],
            xerr=[[raw[i] - lower[i]], [upper[i] - raw[i]]],
            fmt="o",
            mfc=color if significant[i] else "white",
            mec=color,
            color=color,
            capsize=3,
            zorder=4,
        )
    ax1.scatter(baseline, y, marker="s", color=PALETTE["null"], label="randomized baseline", zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels(summary["label"])
    ax1.invert_yaxis()
    ax1.set_xlabel("information (bits)")
    ax1.set_title("Predictive contribution from components of the additional source lineage $d'$")
    ax1.grid(axis="x", alpha=0.18)
    ax1.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#0072B2", markerfacecolor="#0072B2", lw=0, label="distinguishable from baseline"),
            Line2D([0], [0], marker="o", color="#777777", markerfacecolor="white", lw=0, label="not distinguishable from baseline"),
            Line2D([0], [0], marker="s", color=PALETTE["null"], markerfacecolor=PALETTE["null"], lw=0, label="randomized baseline"),
        ],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=7.5,
    )
    _savefig(fig1, fig_dir, "figure09_inter_elc_source_contribution")

    target_order = ["development", "microbiome", "life_history", "epigenetic", "ecological"]
    target_labels = [_level_label(level) for level in target_order]
    profiles = {}
    for analysis in ["whole_source_elc_from_d_prime", "dprime_epigenetic_subsystem"]:
        sub = profile[profile["analysis"] == analysis].set_index("target_level").loc[target_order].reset_index()
        profiles[analysis] = sub
    y2 = np.arange(len(target_order))
    fig2, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), constrained_layout=False)
    fig2.subplots_adjust(left=0.17, right=0.98, top=0.82, bottom=0.25, wspace=0.36)
    styles = [
        ("whole_source_elc_from_d_prime", "#0072B2", "complete ELC of $d'$"),
        ("dprime_epigenetic_subsystem", "#CC79A7", "selected epigenetic segment of $d'$"),
    ]
    for panel_index, (ax, (analysis, color, title)) in enumerate(zip(axes, styles)):
        sub = profiles[analysis]
        estimate = sub["raw_estimate_bits"].to_numpy(dtype=float)
        lower = sub["ci_lower_bits"].to_numpy(dtype=float)
        upper = sub["ci_upper_bits"].to_numpy(dtype=float)
        baseline = sub["surrogate_null_mean_bits"].to_numpy(dtype=float)
        significant = sub["generation_preserving_surrogate_p_value"].to_numpy(dtype=float) <= 0.05
        for i, is_sig in enumerate(significant):
            ax.errorbar(
                estimate[i],
                y2[i],
                xerr=[[estimate[i] - lower[i]], [upper[i] - estimate[i]]],
                fmt="o",
                markersize=5.5,
                mfc=color if is_sig else "white",
                mec=color if is_sig else "#777777",
                ecolor=color if is_sig else "#777777",
                capsize=2.5,
                zorder=4,
            )
        ax.scatter(baseline, y2, marker="s", s=30, color=PALETTE["null"], zorder=3)
        ax.set_yticks(y2)
        if panel_index == 0:
            ax.set_yticklabels(target_labels)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.set_xlabel("conditional mutual information (bits)")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)

    fig2.suptitle("Additional-source information across target levels in lineage $d$", y=0.95)
    fig2.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#333333", markerfacecolor="#333333", lw=0, label="estimated conditional mutual information"),
            Line2D([0], [0], marker="s", color=PALETTE["null"], markerfacecolor=PALETTE["null"], lw=0, label="randomized-source mean"),
            Line2D([0], [0], marker="o", color="#777777", markerfacecolor="white", lw=0, label="not distinguishable from baseline"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        frameon=False,
        fontsize=7.0,
    )
    _savefig(fig2, fig_dir, "figure10_inter_elc_target_profiles")
    return ["figure09_inter_elc_source_contribution", "figure10_inter_elc_target_profiles"]


def make_figures(project_root: Path, out_dir: Path) -> list[str]:
    _setup_matplotlib()
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "data" / "figure_source_tables"
    if fig_dir.exists():
        shutil.rmtree(fig_dir)
    if table_dir.exists():
        shutil.rmtree(table_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    names = [
        _plot_multilevel_dynamics(project_root, fig_dir, table_dir),
        _plot_phenotype_reconstruction(project_root, fig_dir, table_dir),
        _plot_architecture(project_root, fig_dir, table_dir),
        *_plot_graph_reconstruction(project_root, fig_dir, table_dir),
        _plot_predictive_contribution(project_root, fig_dir, table_dir),
        _plot_stability(project_root, fig_dir, table_dir),
        *_plot_location(project_root, fig_dir, table_dir),
        _plot_pid(project_root, fig_dir, table_dir),
        _plot_variant_intervention_fisher(project_root, fig_dir, table_dir),
        *_plot_inter_elc_contribution(project_root, fig_dir, table_dir),
    ]
    return names
