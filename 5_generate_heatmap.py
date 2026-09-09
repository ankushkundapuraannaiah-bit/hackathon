"""
Role 6: Heatmap Renderer & Visualizer (5_generate_heatmap.py)
PS ID: SIH26009 - AI/ML & Space Technology to Identify Manganese Reserves
Ministry of Steel

Description:
Transforms the 2D machine learning probability grid (outputs/4_probability_grid.npy)
into a geo-aligned, glowing PNG image overlay (outputs/heatmap_overlay.png).
Maps probability scores to a high-contrast yellow-to-red heat colormap (YlOrRd),
sets background / low-confidence values (< 0.35) to 100% transparent alpha, and blends
high-confidence subterranean manganese anomalies with vivid opacity for GIS map projection.
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.cm as cm

def generate_heatmap(min_threshold=0.35, max_opacity=0.90):
    print("=" * 65)
    print("ROLE 6: HEATMAP RENDERER & VISUALIZER ENGINE")
    print("Project: Subterranean Mineral Detection Using Satellite Imagery & ML")
    print("=" * 65)
    
    grid_path = os.path.join("outputs", "4_probability_grid.npy")
    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"Missing {grid_path}. Run 4_train_model.py first.")
        
    prob_grid = np.load(grid_path).astype(np.float32)
    H, W = prob_grid.shape
    print(f"Loaded probability grid from {grid_path} (Shape: {H}x{W})")
    print(f"Probability range: Min={np.min(prob_grid):.4f}, Mean={np.mean(prob_grid):.4f}, Max={np.max(prob_grid):.4f}")
    
    # 1. Normalize probabilities above threshold
    # Normalize the range [min_threshold, 1.0] -> [0.0, 1.0] for colormap dynamic range
    prob_clipped = np.clip(prob_grid, 0.0, 1.0)
    
    # Colormap: Yellow-Orange-Red (YlOrRd)
    import matplotlib as mpl
    colormap = mpl.colormaps['YlOrRd']
    
    # Map normalized probabilities to RGBA [0.0, 1.0]
    # For values above min_threshold, stretch across colormap
    norm_vals = np.zeros_like(prob_clipped)
    active_mask = (prob_clipped >= min_threshold)
    
    if np.any(active_mask):
        norm_vals[active_mask] = (prob_clipped[active_mask] - min_threshold) / (1.0 - min_threshold + 1e-6)
        
    rgba_floats = colormap(norm_vals) # Shape: (H, W, 4)
    
    # 2. Configure Alpha Channel (Transparency)
    # Below min_threshold: completely transparent (alpha = 0)
    # Above min_threshold: smooth ramp from 0.35 to max_opacity (0.90)
    alpha = np.zeros((H, W), dtype=np.float32)
    alpha[active_mask] = 0.35 + (norm_vals[active_mask] * (max_opacity - 0.35))
    
    rgba_floats[:, :, 3] = alpha
    
    # Convert to 8-bit unsigned integers [0, 255]
    rgba_uint8 = (rgba_floats * 255.0).astype(np.uint8)
    
    # 3. Save Transparent PNG Image
    out_png = os.path.join("outputs", "heatmap_overlay.png")
    img = Image.fromarray(rgba_uint8, mode='RGBA')
    img.save(out_png, format='PNG')
    
    active_pixels = np.sum(active_mask)
    print(f"\n[OUTPUT] Generated transparent heatmap overlay: {out_png}")
    print(f"         Dimensions: {W} x {H} pixels, 32-bit RGBA")
    print(f"         Transparent Background Threshold: < {min_threshold}")
    print(f"         Rendered Anomaly Pixels: {active_pixels:,} ({active_pixels/(H*W)*100:.2f}% of study area)")
    print("=" * 65)

if __name__ == "__main__":
    thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
    generate_heatmap(min_threshold=thresh)
