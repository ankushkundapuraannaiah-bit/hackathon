"""
Role 4: Machine Learning Classifier (4_train_model.py)
PS ID: SIH26009 - AI/ML & Space Technology to Identify Manganese Reserves
Ministry of Steel

Description:
Trains a supervised machine learning classifier (RandomForestClassifier) on the ground-truth 
spectral dataset (outputs/3_training_data.csv). Evaluates performance using cross-validation,
extracts feature importance rankings to explain remote sensing indicators, and generates a
pixel-wise subterranean manganese prospectivity probability grid (0.0 to 1.0) across the entire
satellite study area. Exports outputs/4_trained_model.pkl, outputs/4_probability_grid.npy,
and outputs/model_metrics.json.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

FEATURE_COLS = ['Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'SWIR_Alteration', 'Ferrous_Mn_Index']

def train_classifier():
    print("=" * 65)
    print("ROLE 4: MACHINE LEARNING PROSPECTIVITY CLASSIFIER")
    print("Project: Subterranean Mineral Detection Using Satellite Imagery & ML")
    print("=" * 65)
    
    # 1. Load Training Data
    csv_path = os.path.join("outputs", "3_training_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Run 3_coordinate_mapper.py first.")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} ground truth spectral samples from {csv_path}")
    
    X = df[FEATURE_COLS].values
    y = df['Label'].values
    
    # Handle any potential NaNs in features
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"Features: {FEATURE_COLS}")
    print(f"Class Distribution: {np.sum(y == 1)} Manganese Deposits vs {np.sum(y == 0)} Non-Mine Controls")
    
    # 2. Train Random Forest Model with Hyperparameter Tuning
    print("\nTraining RandomForestClassifier with Stratified Cross-Validation...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    )
    
    # 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=min(5, np.sum(y == 1)), shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    cv_auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    
    print(f"  Cross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"  Cross-Validation ROC-AUC:  {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f})")
    
    # Fit full model
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    
    train_acc = accuracy_score(y, y_pred)
    train_auc = roc_auc_score(y, y_prob)
    print(f"  Full Dataset Training Accuracy: {train_acc:.3f}")
    print(f"  Full Dataset Training ROC-AUC:  {train_auc:.3f}")
    
    # 3. Feature Importance Analysis for Remote Sensing Defense
    importances = clf.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    
    print("\nGeological Remote Sensing Feature Importances:")
    for feat, imp in feat_imp:
        bar = "#" * int(imp * 30)
        print(f"  {feat:<18} : {imp:.4f} | {bar}")
        
    # 4. Save Model
    model_pkl = os.path.join("outputs", "4_trained_model.pkl")
    joblib.dump(clf, model_pkl)
    print(f"\n[OUTPUT] Saved trained model to {model_pkl}")
    
    # 5. Spatial Inference Across Filtered Satellite Grid
    features_npy = os.path.join("outputs", "2_filtered_features.npy")
    if not os.path.exists(features_npy):
        raise FileNotFoundError(f"Missing {features_npy}. Run 2_spectral_filter.py first.")
        
    print(f"\nLoading satellite feature cube from {features_npy}...")
    feat_cube = np.load(features_npy)
    num_ch, H, W = feat_cube.shape
    print(f"Grid dimensions: {H} rows x {W} cols ({H*W:,} pixels)")
    
    # Channels in 2_filtered_features.npy:
    # 0: Red, 1: NIR, 2: SWIR1, 3: SWIR2, 4: NDVI, 5: SWIR_Ratio, 6: Ferrous_Mn, 7: Bare_Mask
    red_g = feat_cube[0].flatten()
    nir_g = feat_cube[1].flatten()
    swir1_g = feat_cube[2].flatten()
    swir2_g = feat_cube[3].flatten()
    ndvi_g = feat_cube[4].flatten()
    
    eps = 1e-6
    # Compute unmasked alteration ratio for ML input
    swir_alt_g = swir1_g / (swir2_g + eps)
    ferrous_mn_g = swir1_g / (nir_g + eps)
    bare_mask_g = feat_cube[7].flatten() > 0.5
    
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
    
    print("Running model.predict_proba() across all grid cells...")
    grid_probs_raw = clf.predict_proba(grid_X)[:, 1]
    
    # Apply vegetation filter:
    # For dense tree canopy pixels (NDVI > 0.40), optical satellite reflectance reflects leaf chlorophyll,
    # not rock regolith. Scale down confidence on dense canopy to emphasize bare ground anomalies.
    grid_probs = grid_probs_raw.copy()
    grid_probs[~bare_mask_g] = grid_probs[~bare_mask_g] * 0.25
    
    prob_2d = grid_probs.reshape(H, W).astype(np.float32)
    
    out_grid_npy = os.path.join("outputs", "4_probability_grid.npy")
    np.save(out_grid_npy, prob_2d)
    print(f"[OUTPUT] Saved 2D probability grid to {out_grid_npy} (Shape: {prob_2d.shape})")
    
    # 6. Summary Statistics & High Prospectivity Zones
    high_prob_count = np.sum(prob_2d >= 0.65)
    med_prob_count = np.sum((prob_2d >= 0.45) & (prob_2d < 0.65))
    low_prob_count = np.sum(prob_2d < 0.45)
    
    area_sq_km = (H * 30.0) * (W * 30.0) / 1e6
    high_prob_area = high_prob_count * (30.0 * 30.0) / 1e6
    
    print("\nManganese Prospectivity Distribution:")
    print(f"  Total Study Area:         {area_sq_km:.2f} km^2 ({H*W:,} pixels)")
    print(f"  High Confidence Targets:  {high_prob_count:,} pixels ({high_prob_area:.2f} km^2, {high_prob_count/(H*W)*100:.2f}%)")
    print(f"  Moderate Potential:       {med_prob_count:,} pixels ({med_prob_count/(H*W)*100:.2f}%)")
    print(f"  Background / Non-Mine:    {low_prob_count:,} pixels ({low_prob_count/(H*W)*100:.2f}%)")
    print(f"  Peak Prospectivity Score: {np.max(prob_2d):.4f}")
    
    metrics = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 150,
        "max_depth": 5,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "cv_roc_auc_mean": float(cv_auc.mean()),
        "cv_roc_auc_std": float(cv_auc.std()),
        "training_accuracy": float(train_acc),
        "training_roc_auc": float(train_auc),
        "feature_importances": {feat: float(imp) for feat, imp in feat_imp},
        "target_area_km2": float(area_sq_km),
        "high_confidence_area_km2": float(high_prob_area),
        "high_confidence_pixel_count": int(high_prob_count),
        "peak_probability": float(np.max(prob_2d)),
        "mean_probability": float(np.mean(prob_2d))
    }
    
    metrics_json = os.path.join("outputs", "model_metrics.json")
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OUTPUT] Saved evaluation metrics to {metrics_json}")
    print("=" * 65)

if __name__ == "__main__":
    train_classifier()
