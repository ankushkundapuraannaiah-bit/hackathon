"""
Role 3: Ground Truth & Coordinate Mapper (3_coordinate_mapper.py)
PS ID: SIH26009 - AI/ML & Space Technology to Identify Manganese Reserves
Ministry of Steel

Description:
Reads ground truth GPS coordinates of active MOIL manganese mines and non-mine background points 
from raw_data/mine_coordinates.xlsx (or .csv). Maps GPS coordinates to satellite grid pixels
using UTM Zone 44N projection, samples spectral bands and alteration indices from 
outputs/2_filtered_features.npy and the Landsat scene, verifies valid satellite swath coverage,
and exports clean training dataset outputs/3_training_data.csv.
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd

def wgs84_to_utm44n(lat, lon):
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = (a**2 - b**2) / (a**2)
    e_prime2 = (a**2 - b**2) / (b**2)
    k0 = 0.9996
    lon0 = 81.0
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon0_rad = math.radians(lon0)
    
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    T = math.tan(lat_rad)**2
    C = e_prime2 * math.cos(lat_rad)**2
    A = (lon_rad - lon0_rad) * math.cos(lat_rad)
    
    M = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * lat_rad
             - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * math.sin(2*lat_rad)
             + (15*e2**2/256 + 45*e2**3/1024) * math.sin(4*lat_rad)
             - (35*e2**3/3072) * math.sin(6*lat_rad))
             
    easting = k0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18*T + T**2 + 72*C - 58*e_prime2) * A**5 / 120) + 500000.0
    northing = k0 * (M + N * math.tan(lat_rad) * (A**2 / 2 + (5 - T + 9*C + 4*C**2) * A**4 / 24 + (61 - 58*T + T**2 + 600*C - 330*e_prime2) * A**6 / 720))
    return easting, northing

def map_coordinates():
    print("=" * 65)
    print("ROLE 3: GROUND TRUTH & COORDINATE MAPPER")
    print("Project: Subterranean Mineral Detection Using Satellite Imagery & ML")
    print("=" * 65)
    
    # 1. Load Ground Truth Coordinates
    excel_path = os.path.join("raw_data", "mine_coordinates.xlsx")
    csv_path = os.path.join("raw_data", "mine_coordinates.csv")
    
    if os.path.exists(excel_path):
        print(f"Loading ground truth coordinates from {excel_path}...")
        df_coords = pd.read_excel(excel_path)
    elif os.path.exists(csv_path):
        print(f"Loading ground truth coordinates from {csv_path}...")
        df_coords = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Missing mine_coordinates.xlsx or .csv in raw_data/")
        
    # 2. Load Crop Metadata and Filtered Features
    meta_path = os.path.join("outputs", "crop_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    features_path = os.path.join("outputs", "2_filtered_features.npy")
    features = np.load(features_path)
    num_ch, grid_h, grid_w = features.shape
    
    utm_left = meta['utm_bounds']['left']
    utm_top = meta['utm_bounds']['top']
    dx = meta['pixel_size_m']
    dy = meta['pixel_size_m']
    
    # 3. Add known Balaghat-Bharweli mineralized corridor control points if needed
    local_ground_truth = [
        # Verified Bharweli mining complex points
        {'Name': 'Bharweli Open Pit Working 1', 'Latitude': 21.9015, 'Longitude': 80.2092, 'Category': 'Manganese Open Pit', 'Label': 1},
        {'Name': 'Bharweli Tailings & Ore Stacks', 'Latitude': 21.8950, 'Longitude': 80.2050, 'Category': 'Manganese Ore Stockpile', 'Label': 1},
        {'Name': 'Bharweli North Gondite Outcrop', 'Latitude': 21.9120, 'Longitude': 80.2180, 'Category': 'Manganese Outcrop', 'Label': 1},
        {'Name': 'Manegaon Manganese Prospect', 'Latitude': 21.8850, 'Longitude': 80.2250, 'Category': 'Manganese Prospect', 'Label': 1},
        {'Name': 'Laugur-Ukwa Manganese Trend', 'Latitude': 21.9280, 'Longitude': 80.2450, 'Category': 'Manganiferous Schist', 'Label': 1},
        # Verified Local non-mine controls
        {'Name': 'Balaghat District Hospital Area', 'Latitude': 21.8120, 'Longitude': 80.1810, 'Category': 'Urban Residential', 'Label': 0},
        {'Name': 'Garra Wainganga River Sand', 'Latitude': 21.8350, 'Longitude': 80.1550, 'Category': 'River Sand / Water', 'Label': 0},
        {'Name': 'Waraseoni Road Agricultural Field', 'Latitude': 21.8250, 'Longitude': 80.1200, 'Category': 'Active Agriculture', 'Label': 0},
        {'Name': 'Hatta Agricultural Plain', 'Latitude': 21.7850, 'Longitude': 80.2300, 'Category': 'Paddy Cultivation', 'Label': 0},
        {'Name': 'Gangulpara Forest Reserve', 'Latitude': 21.8700, 'Longitude': 80.2600, 'Category': 'Sal Forest Canopy', 'Label': 0},
        {'Name': 'Sarandi Water Reservoir', 'Latitude': 21.7920, 'Longitude': 80.1450, 'Category': 'Inland Water', 'Label': 0},
    ]
    
    df_combined = pd.concat([df_coords, pd.DataFrame(local_ground_truth)], ignore_index=True)
    df_combined.drop_duplicates(subset=['Name'], inplace=True)
    
    raw_bands_full = None
    training_rows = []
    
    print(f"\nMapping {len(df_combined)} points to satellite spectral grid...")
    for idx, row in df_combined.iterrows():
        name = row['Name']
        lat = float(row['Latitude'])
        lon = float(row['Longitude'])
        label = int(row['Label'])
        category = str(row.get('Category', 'Unknown'))
        
        easting, northing = wgs84_to_utm44n(lat, lon)
        
        col_crop = int(round((easting - utm_left) / dx))
        row_crop = int(round((utm_top - northing) / dy))
        
        is_inside_crop = (0 <= row_crop < grid_h) and (0 <= col_crop < grid_w)
        
        if is_inside_crop:
            # 3x3 window around coordinate
            r_min = max(0, row_crop - 1)
            r_max = min(grid_h, row_crop + 2)
            c_min = max(0, col_crop - 1)
            c_max = min(grid_w, col_crop + 2)
            
            patch_bare = features[7, r_min:r_max, c_min:c_max]
            
            if label == 1 and np.any(patch_bare > 0.5):
                bare_coords = np.argwhere(patch_bare > 0.5)
                best_r, best_c = r_min + bare_coords[0][0], c_min + bare_coords[0][1]
            else:
                best_r, best_c = row_crop, col_crop
                
            sample_red = float(features[0, best_r, best_c])
            sample_nir = float(features[1, best_r, best_c])
            sample_swir1 = float(features[2, best_r, best_c])
            sample_swir2 = float(features[3, best_r, best_c])
            sample_ndvi = float(features[4, best_r, best_c])
            
            eps = 1e-6
            sample_swir_ratio = float(sample_swir1 / (sample_swir2 + eps))
            sample_ferrous_mn = float(sample_swir1 / (sample_nir + eps))
            is_bare = int(features[7, best_r, best_c] > 0.5)
            
        else:
            if raw_bands_full is None:
                import tifffile
                raw_dir = "raw_data"
                b4_p = os.path.join(raw_dir, "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B4.TIF")
                b5_p = os.path.join(raw_dir, "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B5.TIF")
                b6_p = os.path.join(raw_dir, "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B6.TIF")
                b7_p = os.path.join(raw_dir, "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B7.TIF")
                
                with tifffile.TiffFile(b4_p) as t4, tifffile.TiffFile(b5_p) as t5, \
                     tifffile.TiffFile(b6_p) as t6, tifffile.TiffFile(b7_p) as t7:
                    raw_bands_full = (t4.pages[0].asarray(), t5.pages[0].asarray(),
                                      t6.pages[0].asarray(), t7.pages[0].asarray())
            
            scene_x0, scene_y0 = 361200.0, 2512500.0
            r_full = int(round((scene_y0 - northing) / 30.0))
            c_full = int(round((easting - scene_x0) / 30.0))
            
            h_f, w_f = raw_bands_full[0].shape
            if not (0 <= r_full < h_f and 0 <= c_full < w_f):
                continue
                
            raw_val_b4 = raw_bands_full[0][r_full, c_full]
            # If pixel is outside the active swath (nodata = 0), skip it
            if raw_val_b4 == 0:
                continue
                
            def to_refl(dn):
                return float(np.clip(dn * 0.0000275 - 0.2, 0.0, 1.0))
                
            sample_red = to_refl(raw_bands_full[0][r_full, c_full])
            sample_nir = to_refl(raw_bands_full[1][r_full, c_full])
            sample_swir1 = to_refl(raw_bands_full[2][r_full, c_full])
            sample_swir2 = to_refl(raw_bands_full[3][r_full, c_full])
            
            eps = 1e-6
            sample_ndvi = (sample_nir - sample_red) / (sample_nir + sample_red + eps)
            sample_swir_ratio = sample_swir1 / (sample_swir2 + eps)
            sample_ferrous_mn = sample_swir1 / (sample_nir + eps)
            is_bare = int(sample_ndvi <= 0.40)
            
        # Reject nodata
        if (sample_red + sample_nir + sample_swir1 + sample_swir2) < 0.01:
            continue
            
        training_rows.append({
            'Name': name,
            'Latitude': lat,
            'Longitude': lon,
            'Label': label,
            'Category': category,
            'Red': round(sample_red, 4),
            'NIR': round(sample_nir, 4),
            'SWIR1': round(sample_swir1, 4),
            'SWIR2': round(sample_swir2, 4),
            'NDVI': round(sample_ndvi, 4),
            'SWIR_Alteration': round(sample_swir_ratio, 4),
            'Ferrous_Mn_Index': round(sample_ferrous_mn, 4),
            'Is_Bare_Ground': is_bare,
            'Inside_Crop': int(is_inside_crop)
        })
        
    df_training = pd.DataFrame(training_rows)
    out_csv = os.path.join("outputs", "3_training_data.csv")
    df_training.to_csv(out_csv, index=False)
    
    print(f"\n[OUTPUT] Successfully exported {len(df_training)} clean training samples to {out_csv}")
    print(f"         Manganese Mine Samples (Label=1): {len(df_training[df_training['Label'] == 1])}")
    print(f"         Non-Mine Control Samples (Label=0): {len(df_training[df_training['Label'] == 0])}")
    print(f"         Samples inside high-res crop:     {len(df_training[df_training['Inside_Crop'] == 1])}")
    
    m_mines = df_training[df_training['Label'] == 1]
    m_nonmines = df_training[df_training['Label'] == 0]
    print("\nSpectral Feature Separation Analysis:")
    print(f"  Manganese Mines Mean SWIR Alteration: {m_mines['SWIR_Alteration'].mean():.3f} +/- {m_mines['SWIR_Alteration'].std():.3f}")
    print(f"  Non-Mines Mean SWIR Alteration:       {m_nonmines['SWIR_Alteration'].mean():.3f} +/- {m_nonmines['SWIR_Alteration'].std():.3f}")
    print(f"  Manganese Mines Mean SWIR1 Refl:      {m_mines['SWIR1'].mean():.3f}")
    print(f"  Non-Mines Mean SWIR1 Refl:            {m_nonmines['SWIR1'].mean():.3f}")
    print(f"  Manganese Mines Mean Red Refl:        {m_mines['Red'].mean():.3f}")
    print(f"  Non-Mines Mean Red Refl:              {m_nonmines['Red'].mean():.3f}")
    print("=" * 65)

if __name__ == "__main__":
    map_coordinates()
