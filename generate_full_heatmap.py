import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Load data
grid_path = os.path.join("outputs", "4_probability_grid.npy")
meta_path = os.path.join("outputs", "crop_metadata.json")
mines_path = os.path.join("outputs", "3_training_data.csv")
res_path = os.path.join("outputs", "detected_reservoirs.json")

prob_grid = np.load(grid_path)
with open(meta_path, "r") as f:
    meta = json.load(f)

wgs = meta["wgs84_bounds"]
extent = [wgs["min_lon"], wgs["max_lon"], wgs["min_lat"], wgs["max_lat"]]

df_mines = None
if os.path.exists(mines_path):
    df_mines = pd.read_csv(mines_path)

reservoirs = []
if os.path.exists(res_path):
    with open(res_path, "r") as f:
        reservoirs = json.load(f)

fig, ax = plt.subplots(figsize=(12, 10), dpi=200)

# Custom colormap: Dark navy -> deep teal -> orange -> bright gold -> crimson
colors = [
    (0.0, "#0F172A"),   # Dark Slate Navy (background)
    (0.20, "#1E293B"),
    (0.35, "#334155"),
    (0.40, "#7C2D12"),  # Warm Sienna
    (0.50, "#C2410C"),  # Orange
    (0.65, "#F97316"),  # Bright Orange
    (0.75, "#FACC15"),  # Vivid Yellow
    (1.0, "#DC2626")    # Crimson Red (Peak Ore)
]
cmap = LinearSegmentedColormap.from_list("manganese_glow", colors, N=256)

im = ax.imshow(prob_grid, extent=extent, origin="upper", cmap=cmap, vmin=0.0, vmax=0.90)

# Overlay Mines
if df_mines is not None:
    mines_only = df_mines[df_mines["Label"] == 1]
    # Filter only inside crop bounds
    in_crop = mines_only[
        (mines_only["Latitude"] >= wgs["min_lat"]) & (mines_only["Latitude"] <= wgs["max_lat"]) &
        (mines_only["Longitude"] >= wgs["min_lon"]) & (mines_only["Longitude"] <= wgs["max_lon"])
    ]
    ax.scatter(
        in_crop["Longitude"], in_crop["Latitude"],
        c="#FFFFFF", edgecolors="#DC2626", s=130, marker="^", linewidths=2.5,
        label="MOIL Active Mines / Deposits", zorder=10
    )
    for _, r in in_crop.iterrows():
        ax.annotate(
            r["Name"].replace("MOIL ", ""),
            xy=(r["Longitude"], r["Latitude"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=8.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="#1E293B", ec="#DC2626", lw=1.2, alpha=0.88),
            zorder=12
        )

# Overlay AI Reservoirs
if reservoirs:
    for idx, r in enumerate(reservoirs[:5]):
        if wgs["min_lat"] <= r["latitude"] <= wgs["max_lat"] and wgs["min_lon"] <= r["longitude"] <= wgs["max_lon"]:
            circle = plt.Circle(
                (r["longitude"], r["latitude"]),
                0.006, color="#38BDF8", fill=False, lw=2, linestyle="--", zorder=8
            )
            ax.add_patch(circle)
            ax.annotate(
                f"💎 {r['reservoir_id']} ({r['confidence_percent']:.0f}%)",
                xy=(r["longitude"], r["latitude"]),
                xytext=(-15, -16), textcoords="offset points",
                fontsize=8, fontweight="bold", color="#38BDF8",
                bbox=dict(boxstyle="round,pad=0.2", fc="#0B132B", ec="#38BDF8", lw=1, alpha=0.85),
                zorder=11
            )

ax.set_title("⚡ GeoManganese AI: Subterranean Prospectivity Heatmap\nBalaghat & Bharweli Manganese Mining Belt (Madhya Pradesh, India)",
             fontsize=14, fontweight="bold", color="#F8FAFC", pad=15)
ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold", color="#CBD5E1")
ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold", color="#CBD5E1")

ax.tick_params(colors="#94A3B8", labelsize=9.5)
ax.grid(color="#334155", linestyle=":", linewidth=0.7, alpha=0.7)
ax.set_facecolor("#0F172A")
fig.patch.set_facecolor("#0B1120")

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("Manganese Mineral Prospectivity Probability", fontsize=10, fontweight="bold", color="#CBD5E1", labelpad=10)
cbar.ax.tick_params(colors="#94A3B8", labelsize=9)
cbar.ax.set_facecolor("#0F172A")
cbar.ax.yaxis.set_ticklabels([f"{t:.1f}" for t in cbar.get_ticks()])

# Add custom indicators to colorbar
cbar.ax.axhline(0.35, color="#F59E0B", lw=1.5, ls="--")
cbar.ax.text(1.3, 0.35, "← Threshold (0.35)", va="center", ha="left", color="#F59E0B", fontsize=8, transform=cbar.ax.get_yaxis_transform())
cbar.ax.axhline(0.70, color="#DC2626", lw=1.5, ls="--")
cbar.ax.text(1.3, 0.70, "← High Target (0.70)", va="center", ha="left", color="#DC2626", fontsize=8, transform=cbar.ax.get_yaxis_transform())

# Legend
ax.legend(loc="lower left", facecolor="#1E293B", edgecolor="#475569", labelcolor="white", fontsize=9)

# Save
out_img = os.path.join("outputs", "prospectivity_heatmap.png")
plt.tight_layout()
plt.savefig(out_img, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("Saved heatmap image to:", out_img)
