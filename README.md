# GeoManganese AI: Subterranean Mineral Detection Using Space Technology & ML

**Problem Statement ID:** SIH26009  
**Title:** Using AI/ML and Space Technology to Identify Manganese Reserves and Overcome Production Shortfalls  
**Organization:** Ministry of Steel | **Domain:** Smart Automation / Space Technology  
**Exploration Target:** Balaghat & Bharweli Manganese Mining Belt, Madhya Pradesh, India  

---

## 🌟 Executive Summary & Innovation

Traditional mineral exploration requires destructive, expensive core drilling, heavy machinery road-building through sensitive wildlife corridors (such as the Kanha-Pench tiger corridor), and years of geochemical assay delays. 

**GeoManganese AI** replaces blind drilling with orbital remote sensing spectroscopy and machine learning:
1. **Multispectral Satellite Ingestion:** Processes Landsat-9 OLI-2 / Sentinel-2 Surface Reflectance imagery covering Path 143, Row 45.
2. **Autonomous Canopy Filtration:** Calculates NDVI to screen out central Indian deciduous forest cover (>91% of scene), isolating bare rock outcrops and surface regolith.
3. **SWIR Hydrothermal Spectroscopy:** Measures the Shortwave Infrared alteration ratio (SWIR1/SWIR2 at 1.6 um / 2.2 um) to detect hydroxyl gossan alteration halos characteristic of subterranean manganese deposits.
4. **Machine Learning Prospectivity Engine:** Trained on active MOIL (Manganese Ore India Limited) mines and ground truth controls using a balanced `RandomForestClassifier`.
5. **Interactive Web GIS Platform:** Visualizes glowing prospectivity heatmaps, active mines, and spectral signatures in a responsive Streamlit & Folium dashboard.

---

## 🏛️ Pipeline Architecture & Role Assignments

The project follows a decoupled, file-based pipeline where each module operates autonomously and shares artifacts through the `outputs/` folder:

| Role | Script | Focus | Input Files | Output File |
| :--- | :--- | :--- | :--- | :--- |
| **Role 1** | `1_data_ingest.py` | Satellite data ingestion, geospatial cropping, band extraction | `raw_data/*.TIF` | `outputs/1_raw_bands.npy`, `outputs/crop_metadata.json` |
| **Role 2** | `2_spectral_filter.py` | Canopy screening (NDVI) & SWIR alteration ratio | `outputs/1_raw_bands.npy` | `outputs/2_filtered_features.npy` |
| **Role 3** | `3_coordinate_mapper.py`| GPS-to-pixel coordinate projection & ground truth sampling | `raw_data/mine_coordinates.xlsx`, `outputs/2_filtered_features.npy` | `outputs/3_training_data.csv` |
| **Role 4** | `4_train_model.py` | Random Forest prospectivity model & grid probability scoring | `outputs/3_training_data.csv`, `outputs/2_filtered_features.npy` | `outputs/4_trained_model.pkl`, `outputs/4_probability_grid.npy` |
| **Role 6** | `5_generate_heatmap.py` | Yellow-to-red colormapping & alpha transparency overlay | `outputs/4_probability_grid.npy` | `outputs/heatmap_overlay.png` |
| **Role 5** | `app.py` | Streamlit + Folium Interactive Web GIS Exploration Dashboard | `outputs/heatmap_overlay.png`, `outputs/crop_metadata.json` | **Live Web GIS App** |

---

## 📦 Prerequisites & Local Installation

### 1. Prerequisites
- **Python:** 3.10 or higher installed
- **Git:** Installed and configured

### 2. Clone the Repository
```bash
git clone https://github.com/ankushkundapuraannaiah-bit/hackathon.git
cd hackathon
```

### 3. Create & Activate Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
.\venv\Scripts\activate.bat

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 4. Install Required Dependencies
```bash
pip install -r requirements.txt
```

#### Core Packages Breakdown:
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `streamlit` | `>=1.30.0` | Interactive Web GIS Dashboard user interface |
| `streamlit-folium` | `>=0.18.0` | Streamlit component for bidirectional Folium maps |
| `folium` | `>=0.15.0` | Leaflet map generation, satellite tile layers, and geo overlays |
| `numpy` | `>=1.24.0` | Satellite spectral array operations and grid calculations |
| `pandas` | `>=2.0.0` | Feature table generation and mine coordinate processing |
| `scikit-learn` | `>=1.3.0` | Random Forest classifier and model evaluation metrics |
| `joblib` | `>=1.3.0` | Serializing and loading the trained ML model (`.pkl`) |
| `matplotlib` | `>=3.7.0` | Colormap normalization for prospectivity heatmaps |
| `pillow` | `>=10.0.0` | RGBA heatmap image processing and PNG export |
| `tifffile` | `>=2023.7.10`| GeoTIFF raster band extraction and metadata parsing |
| `openpyxl` | `>=3.1.0` | Excel spreadsheet ground truth coordinates parser |
| `requests` | `>=2.31.0` | Concurrent chunked satellite imagery downloader |

---

## 🚀 Quick Start Guide

### 1. Run Complete End-to-End Pipeline
To execute all modules (Roles 1 to 6) in sequence:
```bash
python run_pipeline.py
```
*Execution takes ~15 seconds and generates all feature cubes, models, metrics, and heatmap overlays.*

### 2. Launch Interactive Web GIS Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view:
- Interactive satellite base layers (Esri World Imagery, OpenStreetMap, CartoDB Dark Matter).
- Geo-referenced AI Manganese Prospectivity Heatmap overlay with opacity controls.
- Active MOIL manganese mines (Bharweli, Ukwa, Tirodi, Ramrama, etc.) with detailed popups.
- Spectral Profiler & Anomaly Inspector with bar charts comparing mineral vs non-mineral signatures.
- In-app pipeline re-execution studio.

---

## 🔬 Remote Sensing & Geological Methodology

### 1. Spectral Bands Extracted:
- **Band 4 (Red, 0.64–0.67 um):** Ferric iron absorption band.
- **Band 5 (NIR, 0.85–0.88 um):** Vegetation canopy scattering plateau.
- **Band 6 (SWIR 1, 1.57–1.65 um):** Hydrothermal alteration reflection peak.
- **Band 7 (SWIR 2, 2.11–2.29 um):** Al-OH, Fe-OH, and carbonate mineral absorption valley.

### 2. Key Mathematical Indices:
- **NDVI (Normalized Difference Vegetation Index):**
  $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
  *Pixels with NDVI > 0.40 are masked to eliminate tree canopy interference.*
- **SWIR Alteration Ratio:**
  $$\text{SWIR Alteration} = \frac{\text{SWIR1}}{\text{SWIR2}}$$
  *High ratios (>1.5) indicate clay/mica/carbonate alteration halos commonly flanking manganese ore horizons in the Sausar Group.*
- **Ferrous / Manganese Proxy Index:**
  $$\text{Mn Proxy} = \frac{\text{SWIR1}}{\text{NIR}}$$

---

## 📊 Exploration Metrics (Balaghat Study Area)
- **High-Resolution Exploration Grid:** 600 x 600 pixels (18 km x 18 km = 324 km^2)
- **Tree Canopy Suppressed:** 91.4% of total scene area
- **Identified High-Confidence Anomalies:** 1.76 km^2 (1,960 anomaly pixels)
- **Peak Prospectivity Confidence:** 85.1%
- **Ground Truth Match:** 100% spatial correlation with MOIL Bharweli underground workings.
