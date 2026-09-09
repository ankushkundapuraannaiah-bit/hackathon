"""
Master Pipeline Runner: run_pipeline.py
PS ID: SIH26009 - Subterranean Mineral Detection Using AI/ML and Space Technology
Ministry of Steel | Domain: Smart Automation / Space Technology

Orchestrates all 5 pipeline scripts in sequential order:
  Step 1: 1_data_ingest.py       -> outputs/1_raw_bands.npy, outputs/crop_metadata.json
  Step 2: 2_spectral_filter.py   -> outputs/2_filtered_features.npy
  Step 3: 3_coordinate_mapper.py -> outputs/3_training_data.csv
  Step 4: 4_train_model.py       -> outputs/4_trained_model.pkl, outputs/4_probability_grid.npy
  Step 5: 5_generate_heatmap.py  -> outputs/heatmap_overlay.png
"""

import sys
import time
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_step(script_name, description, *args):
    cmd = [sys.executable, script_name] + list(args)
    print(f"\n>>> Running [{script_name}]: {description}...")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[ERROR] Step {script_name} failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[OK] [{script_name}] finished successfully in {elapsed:.2f} seconds.")

def main():
    print("=" * 70)
    print("GEOMANGANESE AI: SPACE TECH & AI EXPLORATION PIPELINE")
    print("Problem Statement ID: SIH26009 | Ministry of Steel")
    print("Exploration Target: Balaghat & Bharweli Manganese Ore Belt")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1: Satellite Data Ingestion
    run_step("1_data_ingest.py", "Role 1 - Satellite Ingestion & Spatial Cropping", "600")
    
    # Step 2: Spectral & Vegetation Filter
    run_step("2_spectral_filter.py", "Role 2 - NDVI Canopy Masking & SWIR Alteration Ratio", "0.40")
    
    # Step 3: Ground Truth Coordinate Mapper
    run_step("3_coordinate_mapper.py", "Role 3 - GPS Coordinate to Pixel Spectral Sampling")
    
    # Step 4: Machine Learning Classifier
    run_step("4_train_model.py", "Role 4 - Random Forest Training & Grid Probability Prediction")
    
    # Step 5: Heatmap Renderer & Visualizer
    run_step("5_generate_heatmap.py", "Role 6 - RGBA Transparent Heatmap Overlay Rendering", "0.35")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"SUCCESS: PIPELINE EXECUTION COMPLETE in {total_time:.2f} seconds!")
    print("All outputs generated in outputs/:")
    print("  - outputs/1_raw_bands.npy")
    print("  - outputs/crop_metadata.json")
    print("  - outputs/2_filtered_features.npy")
    print("  - outputs/3_training_data.csv")
    print("  - outputs/4_trained_model.pkl")
    print("  - outputs/4_probability_grid.npy")
    print("  - outputs/model_metrics.json")
    print("  - outputs/heatmap_overlay.png")
    print("\nTo launch the Web GIS Dashboard, run:")
    print("  streamlit run app.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
