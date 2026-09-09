"""
reservoir_engine.py - Core Analysis Engine for Dataset Ingestion & Manganese Reservoir Detection
PS ID: SIH26009 - AI/ML & Space Technology to Identify Manganese Reserves
Ministry of Steel | Exploration Target: Central Indian Manganese Belt

Provides end-to-end analytical functions to:
1. Ingest new satellite datasets (Preset mining belts, custom lat/lon crops, or uploaded GeoTIFFs).
2. Compute NDVI canopy masks and SWIR hydrothermal alteration spectroscopy.
3. Run machine learning inference across the prospectivity feature cube.
4. Detect, segment, and delineate subterranean manganese reservoirs via connected-component analysis.
5. Render geo-referenced transparent RGBA heatmap overlays and export reservoir catalogs (JSON, CSV, GeoJSON).
"""

import os
import sys
import json
import math
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib as mpl
from scipy.ndimage import label, center_of_mass

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# -------------------------------------------------------------
# Preset Exploration Mining Belts (Central India Sausar Belt)
# -------------------------------------------------------------
PRESET_BELTS = {
    "balaghat_central": {
        "name": "Balaghat & Bharweli Central Belt",
        "description": "Heart of the Balaghat manganese belt hosting Bharweli, the deepest underground manganese mine in Asia.",
        "lat": 21.8988,
        "lon": 80.2078,
        "crop_size": 600,
        "district": "Balaghat, Madhya Pradesh"
    },
    "tirodi_sitapatore": {
        "name": "Tirodi - Sitapatore - Sukli Belt",
        "description": "Historic high-grade braunite and gondite manganese deposits in western Balaghat district.",
        "lat": 21.6883,
        "lon": 79.7042,
        "crop_size": 600,
        "district": "Balaghat, Madhya Pradesh"
    },
    "ukwa_laugur": {
        "name": "Ukwa - Laugur Ore Horizon",
        "description": "Extensive 5 km continuous stratigraphic manganese ore horizon hosted in phyllites and schists.",
        "lat": 21.9667,
        "lon": 80.4667,
        "crop_size": 600,
        "district": "Balaghat, Madhya Pradesh"
    },
    "ramrama_miragpur": {
        "name": "Ramrama - Miragpur Ore Horizon",
        "description": "Prominent open cast and underground manganese mines with high manganese-to-iron ratios.",
        "lat": 21.8500,
        "lon": 79.9167,
        "crop_size": 600,
        "district": "Balaghat, Madhya Pradesh"
    },
    "dongri_buzurg": {
        "name": "Dongri Buzurg - Chikla Belt",
        "description": "Major pyrolusite and cryptomelane manganese dioxide deposits across the MP-Maharashtra border.",
        "lat": 21.5500,
        "lon": 79.6833,
        "crop_size": 600,
        "district": "Bhandara, Maharashtra / MP Border"
    }
}

# -------------------------------------------------------------
# Coordinate Transformation Helpers (WGS84 <-> UTM Zone 44N)
# -------------------------------------------------------------
def wgs84_to_utm44n(lat, lon):
    """Convert WGS84 (lat, lon) to UTM Zone 44N (easting, northing)."""
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = (a**2 - b**2) / (a**2)
    e_prime2 = (a**2 - b**2) / (b**2)
    k0 = 0.9996
    lon0 = 81.0  # central meridian for UTM Zone 44
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

def utm44n_to_wgs84(easting, northing):
    """Convert UTM Zone 44N (easting, northing) to WGS84 (lat, lon)."""
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = (a**2 - b**2) / (a**2)
    e_prime2 = (a**2 - b**2) / (b**2)
    k0 = 0.9996
    lon0 = 81.0
    
    x = easting - 500000.0
    y = northing
    M = y / k0
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    J1 = 3*e1/2 - 27*e1**3/32
    J2 = 21*e1**2/16 - 55*e1**4/32
    J3 = 151*e1**3/96
    J4 = 1097*e1**4/512
    
    fp = mu + J1*math.sin(2*mu) + J2*math.sin(4*mu) + J3*math.sin(6*mu) + J4*math.sin(8*mu)
    C1 = e_prime2 * math.cos(fp)**2
    T1 = math.tan(fp)**2
    R1 = a * (1 - e2) / (1 - e2 * math.sin(fp)**2)**1.5
    N1 = a / math.sqrt(1 - e2 * math.sin(fp)**2)
    D = x / (N1 * k0)
    
    lat = fp - (N1 * math.tan(fp) / R1) * (D**2/2 - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e_prime2) * D**4/24 + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*e_prime2 - 3*C1**2) * D**6/720)
    lon = math.radians(lon0) + (D - (1 + 2*T1 + C1) * D**3/6 + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*e_prime2 + 24*T1**2) * D**5/120) / math.cos(fp)
    return math.degrees(lat), math.degrees(lon)

# -------------------------------------------------------------
# GeoTIFF Loader
# -------------------------------------------------------------
def load_geotiff(filepath):
    """Load GeoTIFF array and geo-transform parameters."""
    import tifffile
    with tifffile.TiffFile(filepath) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        tiepoint = page.tags.get('ModelTiepointTag').value if page.tags.get('ModelTiepointTag') else (0,0,0, 361200.0, 2512500.0, 0.0)
        pixel_scale = page.tags.get('ModelPixelScaleTag').value if page.tags.get('ModelPixelScaleTag') else (30.0, 30.0, 0.0)
        meta = {
            'x0': float(tiepoint[3]),
            'y0': float(tiepoint[4]),
            'dx': float(pixel_scale[0]),
            'dy': float(pixel_scale[1]),
            'crs': 'WGS 84 / UTM zone 44N (EPSG:32644)',
            'width': int(page.shape[1]),
            'height': int(page.shape[0])
        }
        return arr, meta

# -------------------------------------------------------------
# 1. Ingest Data (Spatial Crop from Landsat-9 or Uploaded Bands)
# -------------------------------------------------------------
def ingest_dataset_region(center_lat=21.85, center_lon=80.18, crop_size=600, dataset_name=None):
    """
    Crops a sub-region from the active Landsat-9 OLI-2 scene around given lat/lon.
    Saves outputs/1_raw_bands.npy and outputs/crop_metadata.json.
    """
    raw_dir = "raw_data"
    band_paths = {
        'Red': os.path.join(raw_dir, "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B4.TIF"),
        'NIR': os.path.join(raw_dir, "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B5.TIF"),
        'SWIR1': os.path.join(raw_dir, "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B6.TIF"),
        'SWIR2': os.path.join(raw_dir, "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B7.TIF"),
    }
    for k, p in band_paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required band file: {p}")

    raw_b4, meta = load_geotiff(band_paths['Red'])
    raw_b8, _ = load_geotiff(band_paths['NIR'])
    raw_b11, _ = load_geotiff(band_paths['SWIR1'])
    raw_b12, _ = load_geotiff(band_paths['SWIR2'])

    full_h, full_w = raw_b4.shape

    # Calculate Center Pixel in UTM 44N
    center_e, center_n = wgs84_to_utm44n(center_lat, center_lon)
    center_col = int(round((center_e - meta['x0']) / meta['dx']))
    center_row = int(round((meta['y0'] - center_n) / meta['dy']))

    half = crop_size // 2
    r_start = max(0, center_row - half)
    r_end = min(full_h, r_start + crop_size)
    c_start = max(0, center_col - half)
    c_end = min(full_w, c_start + crop_size)

    if (r_end - r_start) < crop_size:
        r_start = max(0, r_end - crop_size)
    if (c_end - c_start) < crop_size:
        c_start = max(0, c_end - crop_size)

    crop_b4 = raw_b4[r_start:r_end, c_start:c_end].astype(np.float32)
    crop_b8 = raw_b8[r_start:r_end, c_start:c_end].astype(np.float32)
    crop_b11 = raw_b11[r_start:r_end, c_start:c_end].astype(np.float32)
    crop_b12 = raw_b12[r_start:r_end, c_start:c_end].astype(np.float32)

    # Convert USGS L2 SR raw DN to physical reflectance [0.0, 1.0]
    if crop_b4.max() > 100:
        crop_b4 = np.clip(crop_b4 * 0.0000275 - 0.2, 0.0, 1.0)
        crop_b8 = np.clip(crop_b8 * 0.0000275 - 0.2, 0.0, 1.0)
        crop_b11 = np.clip(crop_b11 * 0.0000275 - 0.2, 0.0, 1.0)
        crop_b12 = np.clip(crop_b12 * 0.0000275 - 0.2, 0.0, 1.0)

    raw_bands = np.stack([crop_b4, crop_b8, crop_b11, crop_b12], axis=0)
    output_npy = os.path.join("outputs", "1_raw_bands.npy")
    np.save(output_npy, raw_bands)

    # Calculate geographic bounds
    utm_left = meta['x0'] + c_start * meta['dx']
    utm_right = meta['x0'] + c_end * meta['dx']
    utm_top = meta['y0'] - r_start * meta['dy']
    utm_bottom = meta['y0'] - r_end * meta['dy']

    lat_top, lon_left = utm44n_to_wgs84(utm_left, utm_top)
    lat_bottom, lon_right = utm44n_to_wgs84(utm_right, utm_bottom)

    if dataset_name is None:
        dataset_name = f"Exploration Region ({center_lat:.3f}°N, {center_lon:.3f}°E)"

    metadata = {
        "dataset": dataset_name,
        "region": f"Target Coordinates: {center_lat:.4f}° N, {center_lon:.4f}° E",
        "shape": [int(raw_bands.shape[1]), int(raw_bands.shape[2])],
        "bands": ["B4_Red", "B8_NIR", "B11_SWIR1", "B12_SWIR2"],
        "scene_pixel_offsets": {
            "r_start": int(r_start), "r_end": int(r_end),
            "c_start": int(c_start), "c_end": int(c_end)
        },
        "utm_bounds": {
            "left": float(utm_left), "right": float(utm_right),
            "bottom": float(utm_bottom), "top": float(utm_top)
        },
        "wgs84_bounds": {
            "min_lat": float(min(lat_top, lat_bottom)),
            "max_lat": float(max(lat_top, lat_bottom)),
            "min_lon": float(min(lon_left, lon_right)),
            "max_lon": float(max(lon_left, lon_right))
        },
        "center_coords": {
            "lat": float((lat_top + lat_bottom) / 2),
            "lon": float((lon_left + lon_right) / 2)
        },
        "pixel_size_m": float(meta['dx']),
        "crs": meta['crs']
    }

    meta_json = os.path.join("outputs", "crop_metadata.json")
    with open(meta_json, "w") as f:
        json.dump(metadata, f, indent=2)

    return raw_bands, metadata

def ingest_uploaded_geotiff(file_or_files, bounds=None, dataset_name="Uploaded Dataset"):
    """
    Ingests uploaded GeoTIFF files (either 4 bands or multi-band composite).
    """
    import tifffile
    if isinstance(file_or_files, list) and len(file_or_files) == 4:
        # 4 files passed: [b4, b8, b11, b12]
        bands = []
        for f in file_or_files:
            arr = tifffile.imread(f)
            if arr.ndim > 2:
                arr = arr[0]
            bands.append(arr.astype(np.float32))
        raw_bands = np.stack(bands, axis=0)
    else:
        # Single composite file or first element
        f = file_or_files if not isinstance(file_or_files, list) else file_or_files[0]
        arr = tifffile.imread(f)
        if arr.ndim == 3 and arr.shape[0] >= 4:
            raw_bands = arr[:4].astype(np.float32)
        elif arr.ndim == 3 and arr.shape[2] >= 4:
            raw_bands = np.transpose(arr[:, :, :4], (2, 0, 1)).astype(np.float32)
        else:
            raise ValueError("Uploaded GeoTIFF must contain at least 4 multispectral bands (Red, NIR, SWIR1, SWIR2).")

    # Normalize if DN > 100
    if raw_bands.max() > 100:
        raw_bands = np.clip(raw_bands * 0.0000275 - 0.2, 0.0, 1.0)
    elif raw_bands.max() > 1.0:
        raw_bands = raw_bands / 255.0

    H, W = raw_bands.shape[1], raw_bands.shape[2]
    np.save(os.path.join("outputs", "1_raw_bands.npy"), raw_bands)

    if bounds is None:
        # Default fallback bounds centered on Balaghat
        center_lat, center_lon = 21.8500, 80.1800
        min_lat = center_lat - (H * 30 / 111320) / 2
        max_lat = center_lat + (H * 30 / 111320) / 2
        min_lon = center_lon - (W * 30 / (111320 * math.cos(math.radians(center_lat)))) / 2
        max_lon = center_lon + (W * 30 / (111320 * math.cos(math.radians(center_lat)))) / 2
    else:
        min_lat, max_lat = bounds['min_lat'], bounds['max_lat']
        min_lon, max_lon = bounds['min_lon'], bounds['max_lon']
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

    utm_left, utm_top = wgs84_to_utm44n(max_lat, min_lon)
    utm_right, utm_bottom = wgs84_to_utm44n(min_lat, max_lon)

    metadata = {
        "dataset": dataset_name,
        "region": f"Custom Upload: {H}x{W} pixels",
        "shape": [H, W],
        "bands": ["B4_Red", "B8_NIR", "B11_SWIR1", "B12_SWIR2"],
        "scene_pixel_offsets": {"r_start": 0, "r_end": H, "c_start": 0, "c_end": W},
        "utm_bounds": {
            "left": float(utm_left), "right": float(utm_right),
            "bottom": float(utm_bottom), "top": float(utm_top)
        },
        "wgs84_bounds": {
            "min_lat": float(min_lat), "max_lat": float(max_lat),
            "min_lon": float(min_lon), "max_lon": float(max_lon)
        },
        "center_coords": {
            "lat": float(center_lat), "lon": float(center_lon)
        },
        "pixel_size_m": 30.0,
        "crs": "WGS 84 / UTM zone 44N (EPSG:32644)"
    }

    meta_json = os.path.join("outputs", "crop_metadata.json")
    with open(meta_json, "w") as f:
        json.dump(metadata, f, indent=2)

    return raw_bands, metadata

# -------------------------------------------------------------
# 2. Compute Spectral & Canopy Filter Features
# -------------------------------------------------------------
def compute_spectral_features(raw_bands, ndvi_threshold=0.40):
    """
    Computes NDVI, SWIR Alteration Ratio (SWIR1/SWIR2), Ferrous-Mn Index (SWIR1/NIR),
    and bare ground regolith mask.
    Returns: filtered_features 3D numpy array [8, H, W].
    """
    b4_red = raw_bands[0].astype(np.float32)
    b8_nir = raw_bands[1].astype(np.float32)
    b11_swir1 = raw_bands[2].astype(np.float32)
    b12_swir2 = raw_bands[3].astype(np.float32)

    eps = 1e-6
    # 1. NDVI
    ndvi = (b8_nir - b4_red) / (b8_nir + b4_red + eps)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # 2. Canopy Mask: NDVI > threshold
    canopy_mask = (ndvi > ndvi_threshold)
    bare_ground_mask = ~canopy_mask

    # 3. SWIR Alteration Ratio
    swir_alteration = b11_swir1 / (b12_swir2 + eps)
    swir_alteration_masked = swir_alteration.copy()
    swir_alteration_masked[canopy_mask] = np.nan

    # 4. Secondary Exploration Indices
    ferrous_mn_index = b11_swir1 / (b8_nir + eps)
    ferrous_mn_index[canopy_mask] = np.nan

    # 5. Stack features: [0:Red, 1:NIR, 2:SWIR1, 3:SWIR2, 4:NDVI, 5:SWIR_Ratio, 6:Ferrous_Mn, 7:Bare_Mask]
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

    np.save(os.path.join("outputs", "2_filtered_features.npy"), filtered_features)
    return filtered_features

# -------------------------------------------------------------
# 3. Machine Learning Prospectivity Inference
# -------------------------------------------------------------
def predict_prospectivity(filtered_features, model_path="outputs/4_trained_model.pkl"):
    """
    Applies trained Random Forest Prospectivity Classifier across all grid pixels.
    Returns: 2D probability grid [H, W] with probabilities from 0.0 to 1.0.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Ensure 4_train_model.py has been run.")

    clf = joblib.load(model_path)
    num_ch, H, W = filtered_features.shape

    red_g = filtered_features[0].flatten()
    nir_g = filtered_features[1].flatten()
    swir1_g = filtered_features[2].flatten()
    swir2_g = filtered_features[3].flatten()
    ndvi_g = filtered_features[4].flatten()

    eps = 1e-6
    swir_alt_g = swir1_g / (swir2_g + eps)
    ferrous_mn_g = swir1_g / (nir_g + eps)
    bare_mask_g = filtered_features[7].flatten() > 0.5

    grid_X = np.column_stack([
        red_g,
        nir_g,
        swir1_g,
        swir2_g,
        ndvi_g,
        swir_alt_g,
        ferrous_mn_g
    ])
    grid_X = np.nan_to_num(grid_X, nan=0.0, posinf=10.0, neginf=0.0)

    grid_probs_raw = clf.predict_proba(grid_X)[:, 1]

    # Canopy suppression: downscale dense tree cover to prevent false positives
    grid_probs = grid_probs_raw.copy()
    grid_probs[~bare_mask_g] = grid_probs[~bare_mask_g] * 0.25

    prob_2d = grid_probs.reshape(H, W).astype(np.float32)
    np.save(os.path.join("outputs", "4_probability_grid.npy"), prob_2d)
    return prob_2d

# -------------------------------------------------------------
# 4. Autonomous Manganese Reservoir Detection & Delineation
# -------------------------------------------------------------
def detect_manganese_reservoirs(prob_grid, metadata, filtered_features,
                                confidence_threshold=0.65, min_cluster_pixels=4):
    """
    Segments contiguous high-probability anomaly clusters (subterranean manganese reservoirs),
    computes geo-spatial centroids, surface footprint, peak confidence, and spectral properties.
    Saves outputs/detected_reservoirs.json and outputs/detected_reservoirs.csv.
    Returns: list of reservoir dicts.
    """
    H, W = prob_grid.shape
    utm_left = metadata['utm_bounds']['left']
    utm_top = metadata['utm_bounds']['top']
    dx = metadata['pixel_size_m']
    dy = metadata['pixel_size_m']

    binary_mask = (prob_grid >= confidence_threshold)
    labeled_array, num_features = label(binary_mask)

    reservoirs = []
    red_band = filtered_features[0]
    nir_band = filtered_features[1]
    swir1_band = filtered_features[2]
    swir2_band = filtered_features[3]
    ndvi_band = filtered_features[4]
    swir_alt_raw = swir1_band / (swir2_band + 1e-6)

    for cluster_id in range(1, num_features + 1):
        cmask = (labeled_array == cluster_id)
        pixel_count = int(np.sum(cmask))

        if pixel_count < min_cluster_pixels:
            continue

        # Spatial centroid in pixel space
        r_c, c_c = center_of_mass(cmask)
        
        # Bounding box of cluster
        coords = np.argwhere(cmask)
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)

        # Convert to UTM & WGS84
        utm_cx = utm_left + c_c * dx
        utm_cy = utm_top - r_c * dy
        cen_lat, cen_lon = utm44n_to_wgs84(utm_cx, utm_cy)

        # Bounding box WGS84
        utm_x1 = utm_left + c_min * dx
        utm_y1 = utm_top - r_min * dy
        utm_x2 = utm_left + c_max * dx
        utm_y2 = utm_top - r_max * dy
        lat1, lon1 = utm44n_to_wgs84(utm_x1, utm_y1)
        lat2, lon2 = utm44n_to_wgs84(utm_x2, utm_y2)

        # Metrics
        cluster_probs = prob_grid[cmask]
        peak_prob = float(np.max(cluster_probs))
        mean_prob = float(np.mean(cluster_probs))

        area_sq_m = pixel_count * (dx * dy)
        area_ha = area_sq_m / 10000.0
        area_km2 = area_sq_m / 1e6

        # Equivalent radius in meters
        approx_radius_m = math.sqrt(area_sq_m / math.pi)

        # Spectral characteristics
        mean_swir_alt = float(np.mean(swir_alt_raw[cmask]))
        mean_ndvi = float(np.mean(ndvi_band[cmask]))
        mean_swir1 = float(np.mean(swir1_band[cmask]))
        mean_swir2 = float(np.mean(swir2_band[cmask]))
        mean_red = float(np.mean(red_band[cmask]))
        mean_nir = float(np.mean(nir_band[cmask]))

        # Classification & Priority Tier
        if peak_prob >= 0.75:
            tier = "Tier 1: High-Confidence Primary Manganese Ore Body"
            rec = "Priority 1 Core Drilling Target (MOIL / Geological Survey)"
            color = "#991B1B" # Deep Crimson
        elif peak_prob >= 0.65:
            tier = "Tier 2: Prospective Hydrothermal Alteration Halo"
            rec = "Detailed Surface Trenching & Geophysical Profiling"
            color = "#C2410C" # Rich Orange
        else:
            tier = "Tier 3: Secondary Manganiferous Gossan / Regolith"
            rec = "Reconnaissance Geochemical Soil Sampling"
            color = "#D97706" # Amber

        reservoirs.append({
            "reservoir_id": f"MN-RES-{len(reservoirs)+1:02d}",
            "name": f"Manganese Reservoir {len(reservoirs)+1:02d}",
            "latitude": round(cen_lat, 5),
            "longitude": round(cen_lon, 5),
            "peak_probability": round(peak_prob, 4),
            "mean_probability": round(mean_prob, 4),
            "confidence_percent": round(peak_prob * 100, 1),
            "area_hectares": round(area_ha, 2),
            "area_km2": round(area_km2, 4),
            "approx_radius_meters": round(approx_radius_m, 1),
            "pixel_count": pixel_count,
            "swir_alteration_ratio": round(mean_swir_alt, 3),
            "ndvi": round(mean_ndvi, 3),
            "spectral_reflectance": {
                "Red": round(mean_red, 4),
                "NIR": round(mean_nir, 4),
                "SWIR1": round(mean_swir1, 4),
                "SWIR2": round(mean_swir2, 4)
            },
            "bounds": {
                "min_lat": round(min(lat1, lat2), 5),
                "max_lat": round(max(lat1, lat2), 5),
                "min_lon": round(min(lon1, lon2), 5),
                "max_lon": round(max(lon1, lon2), 5)
            },
            "tier": tier,
            "recommendation": rec,
            "marker_color": color
        })

    # Sort reservoirs by peak probability descending, then by area
    reservoirs.sort(key=lambda x: (x['peak_probability'], x['area_hectares']), reverse=True)

    # Re-assign clean IDs in sorted order
    for idx, r in enumerate(reservoirs):
        r['rank'] = idx + 1
        r['reservoir_id'] = f"MN-RES-{idx+1:02d}"
        r['name'] = f"Manganese Reservoir {idx+1:02d} ({r['tier'].split(':')[0]})"

    # Export to JSON
    json_path = os.path.join("outputs", "detected_reservoirs.json")
    with open(json_path, "w") as f:
        json.dump(reservoirs, f, indent=2)

    # Export to CSV
    csv_path = os.path.join("outputs", "detected_reservoirs.csv")
    csv_rows = []
    for r in reservoirs:
        csv_rows.append({
            "Rank": r['rank'],
            "Reservoir_ID": r['reservoir_id'],
            "Name": r['name'],
            "Latitude": r['latitude'],
            "Longitude": r['longitude'],
            "Confidence_Percent": r['confidence_percent'],
            "Peak_Probability": r['peak_probability'],
            "Mean_Probability": r['mean_probability'],
            "Area_Hectares": r['area_hectares'],
            "Area_Sq_Km": r['area_km2'],
            "Pixel_Count": r['pixel_count'],
            "SWIR_Alteration_Ratio": r['swir_alteration_ratio'],
            "NDVI": r['ndvi'],
            "Classification_Tier": r['tier'],
            "Exploration_Recommendation": r['recommendation']
        })
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    return reservoirs

# -------------------------------------------------------------
# 5. Heatmap Visualizer & PNG Generator
# -------------------------------------------------------------
def render_heatmap_overlay(prob_grid, output_png_path="outputs/heatmap_overlay.png",
                           min_threshold=0.35, max_opacity=0.90):
    """
    Renders high-contrast YlOrRd RGBA PNG overlay with alpha transparency below min_threshold.
    """
    H, W = prob_grid.shape
    prob_clipped = np.clip(prob_grid, 0.0, 1.0)
    colormap = mpl.colormaps['YlOrRd']

    norm_vals = np.zeros_like(prob_clipped)
    active_mask = (prob_clipped >= min_threshold)

    if np.any(active_mask):
        norm_vals[active_mask] = (prob_clipped[active_mask] - min_threshold) / (1.0 - min_threshold + 1e-6)

    rgba_floats = colormap(norm_vals)

    alpha = np.zeros((H, W), dtype=np.float32)
    alpha[active_mask] = 0.35 + (norm_vals[active_mask] * (max_opacity - 0.35))
    rgba_floats[:, :, 3] = alpha

    rgba_uint8 = (rgba_floats * 255.0).astype(np.uint8)
    img = Image.fromarray(rgba_uint8, mode='RGBA')
    img.save(output_png_path, format='PNG')
    return output_png_path

# -------------------------------------------------------------
# 6. Master End-to-End Execution Function
# -------------------------------------------------------------
def run_dataset_analysis(source_type="preset", preset_key="balaghat_central",
                         custom_lat=21.85, custom_lon=80.18, crop_size=600,
                         uploaded_files=None, confidence_threshold=0.65,
                         min_cluster_pixels=4, ndvi_threshold=0.40,
                         progress_callback=None):
    """
    Master pipeline orchestrator for newly added or selected datasets.
    Executes:
    1. Ingestion / Cropping
    2. Spectral & Vegetation Filtering
    3. Machine Learning Prospectivity Inference
    4. Manganese Reservoir Detection & Delineation
    5. Heatmap Rendering & Metrics Updating
    """
    def report_progress(step_text, pct):
        if progress_callback:
            progress_callback(step_text, pct)
        print(f"[{int(pct*100)}%] {step_text}")

    report_progress("Step 1/5: Ingesting satellite multispectral bands...", 0.15)
    if source_type == "preset":
        preset_info = PRESET_BELTS.get(preset_key, PRESET_BELTS["balaghat_central"])
        raw_bands, metadata = ingest_dataset_region(
            center_lat=preset_info["lat"],
            center_lon=preset_info["lon"],
            crop_size=preset_info.get("crop_size", crop_size),
            dataset_name=preset_info["name"]
        )
    elif source_type == "custom_coords":
        raw_bands, metadata = ingest_dataset_region(
            center_lat=custom_lat,
            center_lon=custom_lon,
            crop_size=crop_size,
            dataset_name=f"Custom Target Belt ({custom_lat:.3f}° N, {custom_lon:.3f}° E)"
        )
    elif source_type == "uploaded":
        raw_bands, metadata = ingest_uploaded_geotiff(
            uploaded_files,
            dataset_name="User Uploaded Satellite Imagery"
        )
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    report_progress("Step 2/5: Applying canopy filtration & SWIR alteration spectroscopy...", 0.40)
    features = compute_spectral_features(raw_bands, ndvi_threshold=ndvi_threshold)

    report_progress("Step 3/5: Predicting AI prospectivity probability grid...", 0.65)
    prob_grid = predict_prospectivity(features)

    report_progress("Step 4/5: Delineating subterranean manganese reservoirs...", 0.85)
    reservoirs = detect_manganese_reservoirs(
        prob_grid=prob_grid,
        metadata=metadata,
        filtered_features=features,
        confidence_threshold=confidence_threshold,
        min_cluster_pixels=min_cluster_pixels
    )

    report_progress("Step 5/5: Generating high-resolution heatmap overlay...", 0.95)
    render_heatmap_overlay(prob_grid, min_threshold=0.35)

    # Calculate and save summary metrics
    H, W = prob_grid.shape
    area_sq_km = (H * metadata['pixel_size_m']) * (W * metadata['pixel_size_m']) / 1e6
    high_conf_pixels = int(np.sum(prob_grid >= confidence_threshold))
    high_conf_area = high_conf_pixels * (metadata['pixel_size_m']**2) / 1e6
    total_reservoir_ha = sum(r['area_hectares'] for r in reservoirs)

    metrics = {
        "dataset_name": metadata['dataset'],
        "target_area_km2": float(area_sq_km),
        "high_confidence_area_km2": float(high_conf_area),
        "high_confidence_pixel_count": high_conf_pixels,
        "peak_probability": float(np.max(prob_grid)),
        "mean_probability": float(np.mean(prob_grid)),
        "manganese_reservoirs_detected": len(reservoirs),
        "total_reservoir_hectares": float(total_reservoir_ha),
        "confidence_threshold_used": float(confidence_threshold)
    }
    with open(os.path.join("outputs", "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    report_progress("Analysis Complete! Subterranean manganese reservoirs identified.", 1.0)
    return {
        "status": "success",
        "metadata": metadata,
        "metrics": metrics,
        "reservoirs": reservoirs,
        "total_reservoirs": len(reservoirs)
    }

if __name__ == "__main__":
    print("Testing reservoir_engine execution...")
    result = run_dataset_analysis(source_type="preset", preset_key="balaghat_central")
    print(f"Identified {result['total_reservoirs']} manganese reservoirs!")
    for r in result['reservoirs'][:3]:
        print(f" - {r['reservoir_id']}: {r['confidence_percent']}% confidence, {r['area_hectares']} ha at ({r['latitude']}, {r['longitude']})")
