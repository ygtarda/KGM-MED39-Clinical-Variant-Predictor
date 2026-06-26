#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KGM_MED39 — Yardımcı Fonksiyonlar
Loglama, zamanlama, metrik hesaplama gibi ortak yardımcı araçlar.
"""

import time
import numpy as np
from sklearn.metrics import (
    f1_score, matthews_corrcoef, precision_score, recall_score,
    average_precision_score
)


def log(msg, level="INFO"):
    """Formatlı loglama."""
    simge = {
        "INFO": "ℹ️", "OK": "✅", "WARN": "⚠️",
        "ERR": "❌", "STEP": "🔷", "STAR": "⭐"
    }
    prefix = simge.get(level, "  ")
    print(f" {prefix}  {msg}")


def zaman_olc(func):
    """Fonksiyon süresini ölçen decorator."""
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        dt = time.time() - t0
        log(f"{func.__name__} tamamlandi ({dt:.1f}s)", "OK")
        return result
    return wrapper


def optimal_threshold_bul(y_true, y_probs, metrik='f1_mcc'):
    """
    En iyi karar eşiğini bulur.

    Metrik seçenekleri:
      - 'f1':     Sadece F1 Score maksimize
      - 'mcc':    Sadece MCC maksimize
      - 'f1_mcc': F1 ve MCC'nin ağırlıklı ortalaması (varsayılan)
                  Şartname hem F1 hem MCC istiyor, her ikisini de
                  optimize eden threshold daha dengeli sonuç verir.

    Returns:
        float: Optimal threshold değeri
    """
    best_score = -1
    best_thresh = 0.5

    for thresh in np.arange(0.15, 0.85, 0.005):
        preds = (y_probs >= thresh).astype(int)

        if preds.sum() == 0 or preds.sum() == len(preds):
            continue

        if metrik == 'f1':
            score = f1_score(y_true, preds, zero_division=0)
        elif metrik == 'mcc':
            score = matthews_corrcoef(y_true, preds)
        elif metrik == 'f1_mcc':
            f1 = f1_score(y_true, preds, zero_division=0)
            mcc = matthews_corrcoef(y_true, preds)
            # Normalize MCC to [0,1] range, then combine
            score = 0.6 * f1 + 0.4 * ((mcc + 1) / 2)
        else:
            score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh


def metrik_hesapla(y_true, y_probs, threshold):
    """Tüm metrikleri tek seferde hesaplar."""
    preds = (y_probs >= threshold).astype(int)
    return {
        'f1': f1_score(y_true, preds, zero_division=0),
        'mcc': matthews_corrcoef(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'pr_auc': average_precision_score(y_true, y_probs),
        'threshold': threshold,
    }
