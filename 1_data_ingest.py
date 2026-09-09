"""
Role 1: Satellite Data Ingestion (1_data_ingest.py)
PS ID: SIH26009 - AI/ML & Space Technology to Identify Manganese Reserves
Ministry of Steel

Description:
Opens raw satellite imagery of Balaghat (Landsat 9 / Sentinel-2), crops to a 
focused exploration bounding box encompassing active MOIL manganese mines,
extracts Red, NIR, SWIR1, and SWIR2 bands, and saves them to outputs/1_raw_bands.npy.
Also exports outputs/crop_metadata.json for precise GIS alignment.
"""

import os
import sys
import json
import math
import numpy as np

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# -------------------------------------------------------------
# Coordinate Transformation Helpers (WGS84 <-> UTM Zone 44N)
# -------------------------------------------------------------
def wgs84_to_utm44n(lat, lon):
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = (a**2 - b**2) / (a**2)
    e_prime2 = (a**2 - b**2) / (b**2)
    k0 = 0.9996
    lon0 = 81.0  # central meridian for zone 44
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
# GeoTIFF Loader (Supports rasterio and tifffile)
# -------------------------------------------------------------
def load_geotiff(filepath):
    """Load GeoTIFF array and geo-transform parameters."""
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            arr = src.read(1)
            transform = src.transform
            bounds = src.bounds
            crs = str(src.crs)
            meta = {
                'x0': bounds.left,
                'y0': bounds.top,
                'dx': transform[0],
                'dy': -transform[4],
                'crs': crs,
                'width': src.width,
                'height': src.height
            }
            return arr, meta
    except (ImportError, Exception):
        import tifffile
        with tifffile.TiffFile(filepath) as tif:
            page = tif.pages[0]
            arr = page.asarray()
            tiepoint = page.tags.get('ModelTiepointTag').value
            pixel_scale = page.tags.get('ModelPixelScaleTag').value
            meta = {
                'x0': tiepoint[3],
                'y0': tiepoint[4],
                'dx': pixel_scale[0],
                'dy': pixel_scale[1],
                'crs': 'WGS 84 / UTM zone 44N (EPSG:32644)',
                'width': page.shape[1],
                'height': page.shape[0]
            }
            return arr, meta

def ingest_data(crop_size=600, center_lat=21.85, center_lon=80.18):
    print("=" * 65)
    print("ROLE 1: SATELLITE DATA INGESTION ENGINE")
    print("Project: Subterranean Mineral Detection Using Satellite Imagery & ML")
    print("=" * 65)
    
    raw_dir = "raw_data"
    
    # Check for single Sentinel composite or individual Landsat 9 bands
    composite_candidates = [
        os.path.join(raw_dir, "sentinel_balaghat.tif"),
        os.path.join(raw_dir, "balaghat_multispectral.tif")
    ]
    
    composite_file = None
    for c in composite_candidates:
        if os.path.exists(c):
            composite_file = c
            break
            
    band_paths = {
        'Red': os.path.join(raw_dir, "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B4.TIF"),
        'NIR': os.path.join(raw_dir, "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B5.TIF"),
        'SWIR1': os.path.join(raw_dir, "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B6.TIF"),
        'SWIR2': os.path.join(raw_dir, "LC09_L2SP_143045_20241106_20241107_02_T1_SR_B7.TIF"),
    }
    
    if composite_file and os.path.exists(composite_file):
        print(f"Loading multispectral composite: {composite_file}")
        try:
            import rasterio
            with rasterio.open(composite_file) as src:
                # Assuming bands 4, 8, 11, 12 or 1, 2, 3, 4
                raw_b4 = src.read(1)
                raw_b8 = src.read(2)
                raw_b11 = src.read(3)
                raw_b12 = src.read(4)
                meta = {'x0': src.bounds.left, 'y0': src.bounds.top, 'dx': src.res[0], 'dy': src.res[1]}
        except Exception:
            import tifffile
            with tifffile.TiffFile(composite_file) as tif:
                raw_arr = tif.pages[0].asarray()
                raw_b4, raw_b8, raw_b11, raw_b12 = raw_arr[0], raw_arr[1], raw_arr[2], raw_arr[3]
                meta = {'x0': 361200.0, 'y0': 2512500.0, 'dx': 30.0, 'dy': 30.0}
    else:
        print("Loading Landsat-9 OLI-2 Multispectral Bands from raw_data/:")
        print("  - Band 4 (Red, 0.64-0.67 um)   -> Sentinel-2 Band 4 equivalent")
        print("  - Band 5 (NIR, 0.85-0.88 um)   -> Sentinel-2 Band 8 equivalent")
        print("  - Band 6 (SWIR 1, 1.57-1.65 um)-> Sentinel-2 Band 11 equivalent")
        print("  - Band 7 (SWIR 2, 2.11-2.29 um)-> Sentinel-2 Band 12 equivalent")
        
        for k, p in band_paths.items():
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required band file: {p}")
                
        raw_b4, meta = load_geotiff(band_paths['Red'])
        raw_b8, _ = load_geotiff(band_paths['NIR'])
        raw_b11, _ = load_geotiff(band_paths['SWIR1'])
        raw_b12, _ = load_geotiff(band_paths['SWIR2'])

    full_h, full_w = raw_b4.shape
    print(f"Full Scene Dimensions: {full_h} rows x {full_w} cols (Resolution: {meta['dx']}m)")
    
    # Calculate Center Pixel in UTM 44N
    center_e, center_n = wgs84_to_utm44n(center_lat, center_lon)
    center_col = int(round((center_e - meta['x0']) / meta['dx']))
    center_row = int(round((meta['y0'] - center_n) / meta['dy']))
    
    half = crop_size // 2
    r_start = max(0, center_row - half)
    r_end = min(full_h, r_start + crop_size)
    c_start = max(0, center_col - half)
    c_end = min(full_w, c_start + crop_size)
    
    # Adjust in case near boundaries
    if (r_end - r_start) < crop_size:
        r_start = max(0, r_end - crop_size)
    if (c_end - c_start) < crop_size:
        c_start = max(0, c_end - crop_size)
        
    print(f"\nCropping exploration zone centered at {center_lat} N, {center_lon} E (Balaghat & Bharweli Belt):")
    print(f"  Row Range: [{r_start} : {r_end}], Col Range: [{c_start} : {c_end}]")
    print(f"  Crop Size: {r_end - r_start} x {c_end - c_start} pixels (~{(r_end - r_start)*30/1000:.1f} km x {(c_end - c_start)*30/1000:.1f} km)")
    
    crop_b4 = raw_b4[r_start:r_end, c_start:c_end].astype(np.float32)
    crop_b8 = raw_b8[r_start:r_end, c_start:c_end].astype(np.float32)
    crop_b11 = raw_b11[r_start:r_end, c_start:c_end].astype(np.float32)
    crop_b12 = raw_b12[r_start:r_end, c_start:c_end].astype(np.float32)
    
    # Convert Landsat-9 Surface Reflectance integer DNs to physical reflectance [0.0, 1.0]
    # USGS L2 SR formula: Reflectance = DN * 0.0000275 - 0.2
    # If values are raw DN (> 100), apply scaling; clip valid physical range [0.0, 1.0]
    if crop_b4.max() > 100:
        crop_b4 = np.clip(crop_b4 * 0.0000275 - 0.2, 0.0, 1.0)
        crop_b8 = np.clip(crop_b8 * 0.0000275 - 0.2, 0.0, 1.0)
        crop_b11 = np.clip(crop_b11 * 0.0000275 - 0.2, 0.0, 1.0)
        crop_b12 = np.clip(crop_b12 * 0.0000275 - 0.2, 0.0, 1.0)
    
    # Stack into 4D array: [4, H, W] -> (Red, NIR, SWIR1, SWIR2)
    raw_bands = np.stack([crop_b4, crop_b8, crop_b11, crop_b12], axis=0)
    
    output_npy = os.path.join("outputs", "1_raw_bands.npy")
    np.save(output_npy, raw_bands)
    print(f"\n[OUTPUT] Saved raw bands to {output_npy} (Shape: {raw_bands.shape}, Dtype: {raw_bands.dtype})")
    
    # Geographic Bounding Box calculation
    utm_left = meta['x0'] + c_start * meta['dx']
    utm_right = meta['x0'] + c_end * meta['dx']
    utm_top = meta['y0'] - r_start * meta['dy']
    utm_bottom = meta['y0'] - r_end * meta['dy']
    
    lat_top, lon_left = utm44n_to_wgs84(utm_left, utm_top)
    lat_bottom, lon_right = utm44n_to_wgs84(utm_right, utm_bottom)
    
    metadata = {
        "dataset": "Landsat-9 OLI-2 / Sentinel-2 Equivalent Surface Reflectance",
        "region": "Balaghat & Bharweli Manganese Mining Belt, MP, India",
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
    print(f"[OUTPUT] Saved spatial metadata to {meta_json}")
    print(f"         WGS84 Bounds: Lat [{metadata['wgs84_bounds']['min_lat']:.4f}, {metadata['wgs84_bounds']['max_lat']:.4f}], Lon [{metadata['wgs84_bounds']['min_lon']:.4f}, {metadata['wgs84_bounds']['max_lon']:.4f}]")
    print("=" * 65)

if __name__ == "__main__":
    crop_size = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    ingest_data(crop_size=crop_size)
