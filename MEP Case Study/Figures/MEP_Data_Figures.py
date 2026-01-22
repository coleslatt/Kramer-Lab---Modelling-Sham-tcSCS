import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load ---
df = pd.read_csv("AUC_results_allTrials.csv")

def pick_col(frame, *candidates):
    for c in candidates:
        if c in frame.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}. Available: {list(frame.columns)}")

col_config = pick_col(df, "Config", "config")
col_block  = pick_col(df, "Block", "block")
col_rta    = pick_col(df, "AUC_RTA", "auc_rta", "AUC_rta", "auc_RTA")
col_lta    = pick_col(df, "AUC_LTA", "auc_lta", "AUC_lta", "auc_LTA")

# -------------------------
# Parse fields (Intensity + Stim)
# -------------------------
def parse_intensity(block_str):
    m = re.match(r"^\s*(\d+)", str(block_str))
    return int(m.group(1)) if m else np.nan

def parse_stim(block_str):
    s = str(block_str).lower()
    return "Stim" if "_stim" in s else "No Stim"

df["Intensity"] = df[col_block].apply(parse_intensity).astype("Int64")
df["Stim"] = df[col_block].apply(parse_stim)

# -------------------------
# Remove Baseline completely
# -------------------------
df = df[df[col_config].astype(str).str.strip().str.lower() != "baseline"].copy()

# -------------------------
# Order configs (Config1, Config2, Config3, ...)
# -------------------------
def config_sort_key(cfg):
    s = str(cfg).strip().lower()
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return 10**9  # non-numeric configs go last

configs = sorted(df[col_config].unique(), key=config_sort_key)

config_label_map = {
    "Config1": "DV-I",
    "Config2": "DM-C",
    "Config3": "DM-R",
}


# Only these intensities per your design
intensities = [63, 76]

# -------------------------
# Encodings requested
# -------------------------
# Distinguish Right vs Left by color
side_color = {
    "RTA": "#4C72B0",  # Right (blue)
    "LTA": "#DD8452",  # Left  (orange)
}

# Stim trials as cross-hatched
stim_hatch = {
    "No Stim": None,
    "Stim": "xx",
}

# -------------------------
# Shared Y scaling across BOTH intensity figures
# -------------------------
vals_all = pd.concat([df[col_rta], df[col_lta]], axis=0).dropna().astype(float).values
ymin, ymax = float(np.min(vals_all)), float(np.max(vals_all))
pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.05 * (abs(ymax) + 1e-12)
ylims = (ymin - pad, ymax + pad)

# -------------------------
# Plot one figure for one intensity (63 or 76)
# Each figure contains BOTH RTA and LTA (side-by-side within each config),
# Stim is hatched, mean is a thick green line, and mean connections are BLACK.
# -------------------------
def plot_intensity(intensity):
    title = "Low Intensity TMS (63%)" if intensity == 63 else "High Intensity TMS (76%)"

    fig, ax = plt.subplots(figsize=(12, 6))

    # Layout:
    # X axis is Config (only)
    # Within each config:
    #   - RTA and LTA separated left/right (by position), and colored
    #   - Within each muscle, No-Stim and Stim separated slightly; Stim hatched
    cfg_step = 1.3
    muscle_offset = {"RTA": +0.22, "LTA": -0.22}          # put Right/Left side-by-side
    cond_offset   = {"No Stim": -0.09, "Stim": +0.09}     # separate No Stim vs Stim within muscle
    box_width = 0.16  # slightly narrower boxes

    box_data = []
    box_positions = []
    box_facecolors = []
    box_hatches = []
    box_means = {}  # (cfg, muscle, stim) -> (x, mean)

    for i, cfg in enumerate(configs):
        cfg_center = i * cfg_step

        for muscle_key, metric_col in [("RTA", col_rta), ("LTA", col_lta)]:
            for stim in ["No Stim", "Stim"]:
                sub = df[
                    (df[col_config] == cfg) &
                    (df["Intensity"] == intensity) &
                    (df["Stim"] == stim)
                ]
                vals = sub[metric_col].dropna().astype(float).values
                if len(vals) == 0:
                    continue

                x = cfg_center + muscle_offset[muscle_key] + cond_offset[stim]

                box_data.append(vals)
                box_positions.append(x)
                box_facecolors.append(side_color[muscle_key])
                box_hatches.append(stim_hatch[stim])

                box_means[(cfg, muscle_key, stim)] = (x, float(np.mean(vals)))

    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=box_width,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        meanprops=dict(linestyle="-", linewidth=2.5, color="green"),
    )

    # Apply facecolor + hatch
    for box, fc, ht in zip(bp["boxes"], box_facecolors, box_hatches):
        box.set_facecolor(fc)
        if ht:
            box.set_hatch(ht)

    # Connect No Stim mean -> Stim mean for each muscle within each config (BLACK lines)
    for cfg in configs:
        for muscle_key in ["RTA", "LTA"]:
            k0 = (cfg, muscle_key, "No Stim")
            k1 = (cfg, muscle_key, "Stim")
            if k0 in box_means and k1 in box_means:
                x0, m0 = box_means[k0]
                x1, m1 = box_means[k1]
                ax.plot([x0, x1], [m0, m1], linewidth=2.0, color="black")

    # X ticks: Config only
    cfg_centers = [i * cfg_step for i in range(len(configs))]
    ax.set_xticks(cfg_centers)
    ax.set_xticklabels([config_label_map.get(str(c), str(c)) for c in configs])

    ax.set_title(title)
    ax.set_ylabel("AUC (V·s)")
    ax.set_ylim(*ylims)
    ax.grid(axis="y", alpha=0.25)

    # Legend: Right vs Left color, Stim hatch, Mean line
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor=side_color["RTA"], label="Right Tibial Anterior"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor=side_color["LTA"], label="Left Tibial Anterior"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                      hatch="xx", label="Stimulation"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                      label="No Stimulation"),
        plt.Line2D([0], [0], linestyle="-", linewidth=2.5, color="green", label="Mean"),
        # plt.Line2D([0], [0], linestyle="-", linewidth=2.0, color="black", label="Mean link (No→Stim)"),
    ]
    ax.legend(handles=handles, title="Encoding", loc="best")

    plt.tight_layout()
    fname = out_dir / f"MEP_AUC_{intensity}pct.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()

    
from pathlib import Path

out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)

# -------------------------
# Produce the two requested figures
# -------------------------
low_tms = plot_intensity(63)  # Low intensity
high_tms = plot_intensity(76)  # High intensity
