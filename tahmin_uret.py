#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KGM_MED39 - Final Test Verisi Tahmin Scripti (v4.0)
Yarışma esnasında verilecek test verilerini eğitilmiş pipeline ile işler.

Kullanım:
    python3 tahmin_uret.py --input veriler/TEST_VERISI.csv --output test_sonuclari.csv
    python3 tahmin_uret.py --input veriler/TEST_VERISI.csv --output test_sonuclari.csv --panel KANSER
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
SONUC_DIR = BASE_DIR / 'sonuclar'

# Modüler FE import
sys.path.insert(0, str(BASE_DIR))
from src.feature_engineering import feature_engineering


def load_pipeline():
    """Gerekli model ve dönüştürücüleri yükler."""
    pipeline_path = SONUC_DIR / 'kgm_med39_pipeline.joblib'
    freq_path = SONUC_DIR / 'freq_maps.joblib'

    if not pipeline_path.exists():
        print(f"HATA: Pipeline bulunamadi! Once kgm_med39_predictor.py calistirilmalidir.")
        sys.exit(1)

    print("Modeller ve dönüştürücüler yükleniyor...")
    pipeline = joblib.load(pipeline_path)

    freq_maps = {}
    if freq_path.exists():
        freq_maps = joblib.load(freq_path)

    version = pipeline.get('version', 'v3.0')
    print(f"Pipeline versiyonu: {version}")
    return pipeline, freq_maps


def predict(input_path, output_path, panel=None):
    """Test verisini yükler, FE uygular, tahmin üretir."""
    print(f"[{input_path}] okunuyor...")
    df = pd.read_csv(input_path)
    print(f"  Boyut: {df.shape}")

    pipeline, freq_maps = load_pipeline()

    # Ortak FE modülünü kullan (DRY prensibi)
    print("Özellik mühendisliği uygulanıyor...")
    df_fe, variant_ids = _apply_fe(df, freq_maps)

    print("Veri formatlanıyor...")
    feature_cols = pipeline['feature_columns']
    X = df_fe.reindex(columns=feature_cols, fill_value=0)

    print("Tahminler üretiliyor...")
    X_imp = pipeline['imputer'].transform(X)
    X_sc = pipeline['scaler'].transform(X_imp)

    # Base model tahminleri
    base_probs = []
    weighted_probs = np.zeros(len(X))
    for name, model in pipeline['models'].items():
        w = pipeline['model_weights'].get(name, 1.0 / len(pipeline['models']))
        p = model.predict_proba(X_sc)[:, 1]
        base_probs.append(p)
        weighted_probs += w * p

    # Meta-learner varsa kullan (v4.0)
    if 'meta_learner' in pipeline and pipeline['meta_learner'] is not None:
        meta_X = np.column_stack(base_probs)
        try:
            probs = pipeline['meta_learner'].predict_proba(meta_X)[:, 1]
            threshold = pipeline.get('meta_threshold', pipeline['threshold'])
            print("  Meta-Learner (Stacking) kullaniliyor")
        except Exception:
            probs = weighted_probs
            threshold = pipeline['threshold']
            print("  Agirlikli ensemble kullaniliyor")
    else:
        probs = weighted_probs
        threshold = pipeline['threshold']

    # Panel-specific threshold (varsa)
    if panel and 'panel_thresholds' in pipeline:
        panel_th = pipeline['panel_thresholds'].get(panel.upper())
        if panel_th:
            threshold = panel_th
            print(f"  Panel-specific threshold kullaniliyor ({panel.upper()}): {threshold:.2f}")

    preds = (probs >= threshold).astype(int)

    if variant_ids is None:
        variant_ids = np.arange(len(preds))

    sonuc_df = pd.DataFrame({
        'Variant_ID': variant_ids,
        'Patojenik_Olasiligi': np.round(probs, 4),
        'Tahmin_Label': preds,
        'Tahmin_Sinif': pd.Series(preds).map({0: 'Benign', 1: 'Pathogenic'})
    })

    sonuc_df.to_csv(output_path, index=False)
    print(f"\n✅ İşlem tamam! {len(sonuc_df)} tahmin {output_path} dosyasına kaydedildi.")
    print(f"   Patojenik: {(preds == 1).sum()} | Benign: {(preds == 0).sum()}")
    print(f"   Threshold: {threshold:.2f}")


def _apply_fe(df, freq_maps):
    """Feature engineering uygula (ortak modül kullanarak)."""
    df_fe, variant_ids, _ = feature_engineering(df, freq_maps=freq_maps)
    if 'Label' in df_fe.columns:
        df_fe = df_fe.drop(columns=['Label'])
    return df_fe, variant_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='KGM_MED39 v4.0 - TEKNOFEST Test Verisi Tahmin Araci')
    parser.add_argument('-i', '--input', required=True,
                        help='Girdi test CSV dosyasi yolu')
    parser.add_argument('-o', '--output', default='test_tahminleri.csv',
                        help='Cikti CSV dosyasi yolu')
    parser.add_argument('-p', '--panel', default=None,
                        choices=['MASTER', 'KANSER', 'PAH', 'CFTR'],
                        help='Panel adi (panel-specific threshold icin)')

    args = parser.parse_args()
    predict(args.input, args.output, args.panel)
