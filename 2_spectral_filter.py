"""
Role 2: Spectral & Vegetation Filter (2_spectral_filter.py)
PS ID: SIH26009 - AI/ML & Space Technology to Identify Manganese Reserves
Ministry of Steel

Description:
Solves the "tree canopy" challenge in remote sensing mineral exploration.
Calculates Normalized Difference Vegetation Index (NDVI) to detect dense tree canopy
and masks them out (NDVI > 0.4). For exposed rock outcrops and bare ground, computes
the SWIR Alteration Ratio (Band 11 / Band 12) and secondary manganese/iron oxide proxy indices.
Saves the engineered multi-band feature cube to outputs/2_filtered_features.npy.
"""

import os
import sys
import numpy as np

def run_spectral_filter(ndvi_threshold=0.40):
    print("=" * 65)
    print("ROLE 2: SPECTRAL & VEGETATION FILTER ENGINE")
    print("Project: Subterranean Mineral Detection Using Satellite Imagery & ML")
    print("=" * 65)
    
    input_npy = os.path.join("outputs", "1_raw_bands.npy")
    if not os.path.exists(input_npy):
        raise FileNotFoundError(f"Missing input file {input_npy}. Run 1_data_ingest.py first.")
        
    raw_bands = np.load(input_npy)
    print(f"Loaded raw bands from {input_npy} with shape: {raw_bands.shape}")
    
    # Extract channels: [B4_Red, B8_NIR, B11_SWIR1, B12_SWIR2]
    b4_red = raw_bands[0].astype(np.float32)
    b8_nir = raw_bands[1].astype(np.float32)
    b11_swir1 = raw_bands[2].astype(np.float32)
    b12_swir2 = raw_bands[3].astype(np.float32)
    
    eps = 1e-6
    
    # 1. Calculate NDVI = (NIR - Red) / (NIR + Red)
    print("\n1. Calculating Normalized Difference Vegetation Index (NDVI)...")
    ndvi = (b8_nir - b4_red) / (b8_nir + b4_red + eps)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    # 2. Vegetation Canopy Mask: NDVI > ndvi_threshold (typically 0.40)
    # True = dense tree canopy, False = bare soil, rocky outcrops, regolith
    canopy_mask = (ndvi > ndvi_threshold)
    bare_ground_mask = ~canopy_mask
    
    total_pixels = ndvi.size
    canopy_count = np.sum(canopy_mask)
    bare_count = np.sum(bare_ground_mask)
    
    print(f"   Vegetation Canopy Threshold: NDVI > {ndvi_threshold}")
    print(f"   Dense Forest Canopy Masked:  {canopy_count:,} pixels ({canopy_count/total_pixels*100:.1f}%)")
    print(f"   Exposed Bare Ground / Rock:  {bare_count:,} pixels ({bare_count/total_pixels*100:.1f}%)")
    
    # 3. Calculate SWIR Alteration Ratio = Band 11 / Band 12 (SWIR1 / SWIR2)
    # Mask canopy pixels as NaN to avoid false spectral alteration signatures
    print("\n2. Computing Mineral & SWIR Alteration Ratios...")
    swir_alteration = b11_swir1 / (b12_swir2 + eps)
    
    # Create masked SWIR alteration where tree canopy is NaN
    swir_alteration_masked = swir_alteration.copy()
    swir_alteration_masked[canopy_mask] = np.nan
    
    # 4. Secondary Exploration Indices
    # - Ferrous Iron / Manganese proxy: SWIR1 / NIR
    # - Clay / Hydroxyl alteration index: (SWIR1 + Red) / NIR
    ferrous_mn_index = b11_swir1 / (b8_nir + eps)
    ferrous_mn_index[canopy_mask] = np.nan
    
    # 5. Stack into Filtered Features Grid:
    # [0: Red, 1: NIR, 2: SWIR1, 3: SWIR2, 4: NDVI, 5: SWIR_Alteration (masked), 6: Ferrous_Mn (masked), 7: Bare_Mask (float)]
    filtered_features = np.stack([
        b4_red,
        b8_nir,
        b11_swir1,
        b12_swir2,
        ndvi,
        swir_alteration_masked,
        ferrous_mn_index,
        bare_ground_mask.astype(np.float32)
    ], axis=0)
    
    output_npy = os.path.join("outputs", "2_filtered_features.npy")
    np.save(output_npy, filtered_features)
    
    print(f"\n[OUTPUT] Successfully saved filtered features to {output_npy}")
    print(f"         Feature Shape: {filtered_features.shape}")
    print(f"         Channels: [0:Red, 1:NIR, 2:SWIR1, 3:SWIR2, 4:NDVI, 5:SWIR_Ratio, 6:Ferrous_Mn, 7:Bare_Mask]")
    
    valid_ratios = swir_alteration_masked[~np.isnan(swir_alteration_masked)]
    if len(valid_ratios) > 0:
        print(f"         Bare Ground SWIR Alteration: Min={np.min(valid_ratios):.3f}, Mean={np.mean(valid_ratios):.3f}, Max={np.max(valid_ratios):.3f}")
    print("=" * 65)

if __name__ == "__main__":
    thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.40
    run_spectral_filter(ndvi_threshold=thresh)
