#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║  KGM_MED39 - Klinik Genomik Varyant Patojenite Tahmin Sistemi    ║
║  TEKNOFEST 2026 - Sağlıkta Yapay Zeka Yarışması                  ║
║  Üniversite ve Üzeri Seviyesi - v4.0 (Championship Edition)      ║
╚══════════════════════════════════════════════════════════════════╝

v4.0 Yenilikler:
  1. Gerçek Klinik Stres Testi Simülasyonu (CV fold resampling)
  2. Stacking Meta-Learner (OOF + LogisticRegression)
  3. Tüm modeller için Optuna Bayesian Optimizasyonu
  4. Gelişmiş Feature Engineering (entropy, polarity, PCA)
  5. F1 + MCC Kompozit Threshold Optimizasyonu
  6. Panel-Specific Threshold Tuning
  7. TabNet Desteği (opsiyonel)
  8. SHAP Açıklanabilirlik (gelişmiş)
  9. Kapsamlı Raporlama

Kritik Şartname Bilgisi (Sayfa 7-8):
  - Eğitim seti: Patojenik baskın (~%73)
  - Test seti: Benign baskın (~%86) → "Klinik Stres Testi"
  - Temel sıralama metriği: F1 Score
  - Ek metrik: MCC (Matthews Korelasyon Katsayısı)

Takım: KGM_MED39 - Kırklareli Üniversitesi
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
import warnings
import time
import copy
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, matthews_corrcoef, average_precision_score,
    confusion_matrix, precision_score, recall_score,
    precision_recall_curve, roc_auc_score
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Modüler import
from src.config import (
    BASE_DIR, VERI_DIR, SONUC_DIR, RANDOM_STATE,
    N_SPLITS, N_REPEATS, MISSING_THRESH_RATIO,
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, TEST_DAGILIM, PANEL_DOSYALARI
)
from src.feature_engineering import feature_engineering
from src.utils import log, zaman_olc, optimal_threshold_bul, metrik_hesapla

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_STATE)


# ══════════════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════════

@zaman_olc
def veri_yukle(dosya_yolu):
    """CSV dosyasını yükler ve label kontrolü yapar."""
    if not os.path.exists(dosya_yolu):
        log(f"HATA: {dosya_yolu} bulunamadi!", "ERR")
        sys.exit(1)
    df = pd.read_csv(dosya_yolu)
    log(f"Veri yuklendi: {df.shape[0]} satir, {df.shape[1]} kolon", "INFO")
    if 'Label' not in df.columns:
        log("HATA: 'Label' kolonu bulunamadi!", "ERR")
        sys.exit(1)
    if df['Label'].dtype == 'object':
        le = LabelEncoder()
        df['Label'] = le.fit_transform(df['Label'].astype(str))
        log("Label string -> integer donusumu yapildi", "WARN")
    unique_labels = sorted(df['Label'].unique())
    log(f"Label degerleri: {unique_labels}", "INFO")
    return df


# ══════════════════════════════════════════════════════════════════
# 2. VERİ HAZIRLAMA
# ══════════════════════════════════════════════════════════════════

@zaman_olc
def veri_hazirla(df, missing_thresh=MISSING_THRESH_RATIO):
    """Eksik sütunları temizle, X ve y ayır."""
    thresh_limit = int(len(df) * (1 - missing_thresh))
    df_temiz = df.dropna(thresh=thresh_limit, axis=1).copy()
    silinen = set(df.columns) - set(df_temiz.columns)
    if silinen:
        log(f"{len(silinen)} kolon eksik veri nedeniyle silindi (esik: %{missing_thresh*100:.0f})", "INFO")
    X = df_temiz.drop(columns=['Label'])
    y = df_temiz['Label']
    b = (y == 0).sum()
    p = (y == 1).sum()
    log(f"Sinif dagilimi -> Benign: {b} ({b/len(y)*100:.1f}%) | Pathogenic: {p} ({p/len(y)*100:.1f}%)", "INFO")
    return X, y


# ══════════════════════════════════════════════════════════════════
# 3. MODEL TANIMLARI
# ══════════════════════════════════════════════════════════════════

def modelleri_olustur(spw, best_params=None):
    """Ensemble için tüm modelleri oluştur."""

    # ─── XGBoost ───
    if best_params and 'xgb' in best_params:
        xp = best_params['xgb']
    else:
        xp = dict(n_estimators=500, learning_rate=0.03, max_depth=6,
                  min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=1.0, gamma=0.1)
    models = {
        'xgb': XGBClassifier(**xp, scale_pos_weight=spw,
                             random_state=RANDOM_STATE,
                             eval_metric='logloss', verbosity=0, n_jobs=-1),
    }

    # ─── ExtraTrees ───
    if best_params and 'et' in best_params:
        ep = best_params['et']
        models['et'] = ExtraTreesClassifier(**ep, class_weight='balanced',
                                            random_state=RANDOM_STATE, n_jobs=-1)
    else:
        models['et'] = ExtraTreesClassifier(
            n_estimators=500, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1)

    # ─── LightGBM ───
    try:
        from lightgbm import LGBMClassifier
        if best_params and 'lgbm' in best_params:
            lp = best_params['lgbm']
        else:
            lp = dict(n_estimators=500, learning_rate=0.03, max_depth=7,
                      num_leaves=63, min_child_samples=20, subsample=0.8,
                      colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
        models['lgbm'] = LGBMClassifier(**lp, scale_pos_weight=spw,
                                        random_state=RANDOM_STATE,
                                        verbose=-1, n_jobs=-1)
    except ImportError:
        log("LightGBM bulunamadi, atlaniyor", "WARN")

    # ─── CatBoost ───
    try:
        from catboost import CatBoostClassifier
        if best_params and 'catboost' in best_params:
            cp = best_params['catboost']
        else:
            cp = dict(iterations=500, learning_rate=0.03, depth=6,
                      l2_leaf_reg=3)
        models['catboost'] = CatBoostClassifier(
            **cp, auto_class_weights='Balanced',
            random_state=RANDOM_STATE, verbose=0)
    except ImportError:
        log("CatBoost bulunamadi, atlaniyor", "WARN")

    return models


# ══════════════════════════════════════════════════════════════════
# 4. OPTUNA OPTİMİZASYON (Tüm Modeller)
# ══════════════════════════════════════════════════════════════════

@zaman_olc
def optuna_optimize(X, y, spw):
    """Optuna Bayesian hiperparametre optimizasyonu — tüm modeller."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log("Optuna yuklu degil, varsayilan parametreler kullanilacak.", "WARN")
        return None

    best_params = {}

    def _cv_evaluate(X, y, model_fn, spw):
        """Ortak CV değerlendirme fonksiyonu."""
        skf = StratifiedKFold(n_splits=3, shuffle=True,
                              random_state=RANDOM_STATE)
        scores = []
        for tr_i, vl_i in skf.split(X, y):
            imp = SimpleImputer(strategy='median')
            sc = RobustScaler()
            Xtr = sc.fit_transform(imp.fit_transform(X.iloc[tr_i]))
            Xvl = sc.transform(imp.transform(X.iloc[vl_i]))
            m = model_fn()
            m.fit(Xtr, y.iloc[tr_i])
            probs = m.predict_proba(Xvl)[:, 1]
            th = optimal_threshold_bul(y.iloc[vl_i], probs, metrik='f1_mcc')
            preds = (probs >= th).astype(int)
            f1 = f1_score(y.iloc[vl_i], preds)
            mcc = matthews_corrcoef(y.iloc[vl_i], preds)
            scores.append(0.6 * f1 + 0.4 * ((mcc + 1) / 2))
        return np.mean(scores)

    # ─── XGBoost Optimizasyonu ───
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
        }
        return _cv_evaluate(X, y, lambda: XGBClassifier(
            **params, scale_pos_weight=spw, random_state=RANDOM_STATE,
            eval_metric='logloss', verbosity=0, n_jobs=-1), spw)

    log("XGBoost hiperparametre optimizasyonu basliyor...", "STEP")
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(xgb_objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)
    best_params['xgb'] = study.best_params
    log(f"XGBoost en iyi skor: {study.best_value:.4f}", "OK")

    # ─── LightGBM Optimizasyonu ───
    try:
        from lightgbm import LGBMClassifier

        def lgbm_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            return _cv_evaluate(X, y, lambda: LGBMClassifier(
                **params, scale_pos_weight=spw, random_state=RANDOM_STATE,
                verbose=-1, n_jobs=-1), spw)

        log("LightGBM hiperparametre optimizasyonu basliyor...", "STEP")
        study2 = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study2.optimize(lgbm_objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT)
        best_params['lgbm'] = study2.best_params
        log(f"LightGBM en iyi skor: {study2.best_value:.4f}", "OK")
    except ImportError:
        pass

    # ─── CatBoost Optimizasyonu ───
    try:
        from catboost import CatBoostClassifier

        def catboost_objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 200, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            }
            return _cv_evaluate(X, y, lambda: CatBoostClassifier(
                **params, auto_class_weights='Balanced',
                random_state=RANDOM_STATE, verbose=0), spw)

        log("CatBoost hiperparametre optimizasyonu basliyor...", "STEP")
        study3 = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study3.optimize(catboost_objective, n_trials=min(OPTUNA_N_TRIALS, 40),
                        timeout=OPTUNA_TIMEOUT)
        best_params['catboost'] = study3.best_params
        log(f"CatBoost en iyi skor: {study3.best_value:.4f}", "OK")
    except ImportError:
        pass

    # ─── ExtraTrees Optimizasyonu ───
    def et_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 6, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }
        return _cv_evaluate(X, y, lambda: ExtraTreesClassifier(
            **params, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1), spw)

    log("ExtraTrees hiperparametre optimizasyonu basliyor...", "STEP")
    study4 = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study4.optimize(et_objective, n_trials=min(OPTUNA_N_TRIALS, 30),
                    timeout=OPTUNA_TIMEOUT // 2)
    best_params['et'] = study4.best_params
    log(f"ExtraTrees en iyi skor: {study4.best_value:.4f}", "OK")

    return best_params


# ══════════════════════════════════════════════════════════════════
# 5. KLİNİK STRES TESTİ SİMÜLASYONU
# ══════════════════════════════════════════════════════════════════

def stres_testi_resample(y_val, target_benign_ratio=0.86):
    """
    Validation fold'un sınıf dağılımını test setine yaklaştırır.
    Şartnameye göre test setinde ~%86 Benign olacak.

    Bu fonksiyon, validation indekslerini resample ederek
    Benign-baskın bir validation set oluşturur.
    """
    idx_benign = np.where(y_val == 0)[0]
    idx_pathogenic = np.where(y_val == 1)[0]

    n_pathogenic = len(idx_pathogenic)
    if n_pathogenic == 0:
        return np.arange(len(y_val))

    # Hedef Benign sayısı: pathogenic_count * (ratio / (1 - ratio))
    n_benign_target = int(n_pathogenic * target_benign_ratio / (1 - target_benign_ratio))
    n_benign_target = max(n_benign_target, len(idx_benign))

    if len(idx_benign) < n_benign_target:
        # Benign yeterli değilse oversampling
        idx_benign_resampled = resample(idx_benign, replace=True,
                                        n_samples=n_benign_target,
                                        random_state=RANDOM_STATE)
    else:
        idx_benign_resampled = idx_benign[:n_benign_target]

    return np.concatenate([idx_pathogenic, idx_benign_resampled])


# ══════════════════════════════════════════════════════════════════
# 6. STACKING ENSEMBLE İLE ÇAPRAZ DOĞRULAMA
# ══════════════════════════════════════════════════════════════════

@zaman_olc
def capraz_dogrulama_stacking(X, y, models, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    """
    Gelişmiş Stacking Ensemble ile Çapraz Doğrulama.

    Level-1: Her base model OOF tahminleri üretir
    Level-2: LogisticRegression meta-learner OOF tahminlerini birleştirir
    Ek: Klinik Stres Testi simülasyonu da raporlanır
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=RANDOM_STATE)
    total_folds = n_splits * n_repeats

    # OOF tahmin matrisleri (stacking için)
    n_models = len(models)
    model_names = list(models.keys())
    oof_predictions = np.zeros((len(y), n_models))
    oof_counts = np.zeros(len(y))

    # Metrik toplama
    all_f1, all_mcc, all_pr_auc = [], [], []
    all_prec, all_rec = [], []
    # Stres testi metrikleri
    stress_f1, stress_mcc = [], []
    fold_detaylari = []

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        Xtr, Xvl = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yvl = y.iloc[train_idx], y.iloc[val_idx]

        # Ön işleme
        imp = SimpleImputer(strategy='median')
        Xtr_imp = imp.fit_transform(Xtr)
        Xvl_imp = imp.transform(Xvl)
        sc = RobustScaler()
        Xtr_sc = sc.fit_transform(Xtr_imp)
        Xvl_sc = sc.transform(Xvl_imp)

        # ─── Level 1: Base model tahminleri ───
        model_probs = []
        for mi, (name, model) in enumerate(models.items()):
            m = copy.deepcopy(model)

            m.fit(Xtr_sc, ytr)

            probs = m.predict_proba(Xvl_sc)[:, 1]
            model_probs.append(probs)

            # OOF predictions (ilk repeat için stacking)
            if fold_idx < n_splits:
                oof_predictions[val_idx, mi] += probs

        oof_counts[val_idx] += 1

        # ─── Ağırlıklı Ensemble ───
        model_weights = []
        for probs in model_probs:
            tmp_f1 = f1_score(yvl, (probs >= 0.5).astype(int), zero_division=0)
            model_weights.append(max(tmp_f1, 0.01))

        weights = np.array(model_weights)
        weights = weights / weights.sum()
        ens_probs = sum(w * p for w, p in zip(weights, model_probs))

        # ─── Normal Threshold (F1+MCC optimizasyonu) ───
        best_th = optimal_threshold_bul(yvl, ens_probs, metrik='f1_mcc')
        preds = (ens_probs >= best_th).astype(int)

        f1 = f1_score(yvl, preds)
        mcc = matthews_corrcoef(yvl, preds)
        pr_auc = average_precision_score(yvl, ens_probs)
        prec = precision_score(yvl, preds, zero_division=0)
        rec = recall_score(yvl, preds, zero_division=0)

        all_f1.append(f1)
        all_mcc.append(mcc)
        all_pr_auc.append(pr_auc)
        all_prec.append(prec)
        all_rec.append(rec)

        # ─── Klinik Stres Testi Simülasyonu ───
        stress_idx = stres_testi_resample(yvl.values, target_benign_ratio=0.86)
        if len(stress_idx) > 0:
            yvl_stress = yvl.values[stress_idx]
            ens_stress = ens_probs[stress_idx]
            th_stress = optimal_threshold_bul(yvl_stress, ens_stress, metrik='f1_mcc')
            preds_stress = (ens_stress >= th_stress).astype(int)
            s_f1 = f1_score(yvl_stress, preds_stress, zero_division=0)
            s_mcc = matthews_corrcoef(yvl_stress, preds_stress)
            stress_f1.append(s_f1)
            stress_mcc.append(s_mcc)

        cm = confusion_matrix(yvl, preds)
        fold_detaylari.append({
            'fold': fold_idx + 1, 'f1': f1, 'pr_auc': pr_auc, 'mcc': mcc,
            'precision': prec, 'recall': rec, 'threshold': best_th,
            'weights': weights.tolist(), 'cm': cm
        })

        if (fold_idx + 1) % n_splits == 0 or fold_idx == 0:
            stress_info = f" | Stres F1: {s_f1:.4f}" if stress_f1 else ""
            log(f"Fold {fold_idx+1}/{total_folds} -> F1: {f1:.4f} | MCC: {mcc:.4f} | "
                f"PR-AUC: {pr_auc:.4f} (th={best_th:.2f}){stress_info}", "INFO")

    # ─── OOF normalize ───
    mask = oof_counts > 0
    for mi in range(n_models):
        oof_predictions[mask, mi] /= oof_counts[mask]

    # ─── Level 2: Meta-Learner (Stacking) ───
    log("Stacking Meta-Learner egitiliyor...", "STEP")
    meta_X = oof_predictions[mask]
    meta_y = y.values[mask]

    meta_learner = LogisticRegression(
        C=1.0, penalty='l2', solver='lbfgs',
        class_weight='balanced', max_iter=1000,
        random_state=RANDOM_STATE
    )
    meta_learner.fit(meta_X, meta_y)

    # Meta-learner ile OOF tahminleri
    meta_probs = meta_learner.predict_proba(meta_X)[:, 1]
    meta_th = optimal_threshold_bul(meta_y, meta_probs, metrik='f1_mcc')
    meta_preds = (meta_probs >= meta_th).astype(int)
    meta_f1 = f1_score(meta_y, meta_preds)
    meta_mcc = matthews_corrcoef(meta_y, meta_preds)

    log(f"Meta-Learner OOF -> F1: {meta_f1:.4f} | MCC: {meta_mcc:.4f} (th={meta_th:.2f})", "STAR")

    # ─── Ağırlıklı ensemble OOF (karşılaştırma için) ───
    avg_weights = np.mean([fd['weights'] for fd in fold_detaylari], axis=0)
    oof_ens = np.zeros(len(y))
    for mi in range(n_models):
        oof_ens[mask] += avg_weights[mi] * oof_predictions[mask, mi]
    global_th = optimal_threshold_bul(y, oof_ens, metrik='f1_mcc')

    return {
        'fold_detaylari': fold_detaylari,
        'ort_f1': np.mean(all_f1), 'std_f1': np.std(all_f1),
        'ort_pr_auc': np.mean(all_pr_auc), 'std_pr_auc': np.std(all_pr_auc),
        'ort_mcc': np.mean(all_mcc), 'std_mcc': np.std(all_mcc),
        'ort_prec': np.mean(all_prec), 'ort_rec': np.mean(all_rec),
        'global_threshold': global_th,
        'oof_probs': oof_ens,
        'meta_learner': meta_learner,
        'meta_threshold': meta_th,
        'meta_f1': meta_f1, 'meta_mcc': meta_mcc,
        'oof_predictions': oof_predictions,
        'model_names': model_names,
        'avg_weights': avg_weights,
        'stress_f1': np.mean(stress_f1) if stress_f1 else None,
        'stress_mcc': np.mean(stress_mcc) if stress_mcc else None,
    }


# ══════════════════════════════════════════════════════════════════
# 7. FİNAL MODEL EĞİTİMİ
# ══════════════════════════════════════════════════════════════════

@zaman_olc
def final_model_egit(X, y, models, sonuclar):
    """
    Tüm veri üzerinde final modeli eğitir.
    Stacking meta-learner dahil pipeline kaydeder.
    """
    threshold = sonuclar['global_threshold']
    meta_learner = sonuclar['meta_learner']
    meta_threshold = sonuclar['meta_threshold']

    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)
    sc = RobustScaler()
    X_sc = sc.fit_transform(X_imp)

    egitilmis = {}
    model_w = []
    model_names = list(models.keys())

    for name, model in models.items():
        m = copy.deepcopy(model)

        m.fit(X_sc, y)
        egitilmis[name] = m

        # OOF ağırlık tahmini
        skf = StratifiedKFold(n_splits=3, shuffle=True,
                              random_state=RANDOM_STATE)
        fs = []
        for tri, vli in skf.split(X, y):
            mt = copy.deepcopy(model)
            impt = SimpleImputer(strategy='median')
            sct = RobustScaler()
            Xt = sct.fit_transform(impt.fit_transform(X.iloc[tri]))
            Xv = sct.transform(impt.transform(X.iloc[vli]))
            mt.fit(Xt, y.iloc[tri])
            pr = mt.predict_proba(Xv)[:, 1]
            pd_pr = (pr >= threshold).astype(int)
            f1 = f1_score(y.iloc[vli], pd_pr, zero_division=0)
            mcc = matthews_corrcoef(y.iloc[vli], pd_pr)
            fs.append(0.6 * f1 + 0.4 * ((mcc + 1) / 2))
        model_w.append(np.mean(fs))

    weights = np.array(model_w)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(egitilmis)) / len(egitilmis)

    # ─── Pipeline kaydet ───
    pipeline = {
        'models': egitilmis,
        'imputer': imp,
        'scaler': sc,
        'threshold': threshold,
        'model_weights': dict(zip(egitilmis.keys(), weights)),
        'feature_columns': X.columns.tolist(),
        'meta_learner': meta_learner,
        'meta_threshold': meta_threshold,
        'version': 'v4.0',
    }

    joblib.dump(pipeline, SONUC_DIR / 'kgm_med39_pipeline.joblib')
    joblib.dump(imp, SONUC_DIR / 'veri_imputer.joblib')
    joblib.dump(sc, SONUC_DIR / 'veri_scaler.joblib')
    joblib.dump(X.columns, SONUC_DIR / 'model_kolonlari.joblib')
    log(f"Pipeline kaydedildi: {SONUC_DIR / 'kgm_med39_pipeline.joblib'}", "OK")
    log(f"Model agirliklari: {dict(zip(egitilmis.keys(), [f'{w:.3f}' for w in weights]))}", "INFO")
    return pipeline


def ensemble_tahmin(pipeline, X_new, use_meta=True):
    """
    Pipeline ile tahmin üret.
    use_meta=True ise stacking meta-learner kullanılır.
    """
    X_imp = pipeline['imputer'].transform(X_new)
    X_sc = pipeline['scaler'].transform(X_imp)

    # Base model tahminleri
    base_probs = []
    weighted_probs = np.zeros(len(X_new))

    for name, model in pipeline['models'].items():
        w = pipeline['model_weights'].get(name, 1.0 / len(pipeline['models']))
        p = model.predict_proba(X_sc)[:, 1]
        base_probs.append(p)
        weighted_probs += w * p

    # Meta-learner varsa kullan
    if use_meta and 'meta_learner' in pipeline and pipeline['meta_learner'] is not None:
        meta_X = np.column_stack(base_probs)
        try:
            probs = pipeline['meta_learner'].predict_proba(meta_X)[:, 1]
            threshold = pipeline.get('meta_threshold', pipeline['threshold'])
        except Exception:
            probs = weighted_probs
            threshold = pipeline['threshold']
    else:
        probs = weighted_probs
        threshold = pipeline['threshold']

    preds = (probs >= threshold).astype(int)
    return probs, preds


# ══════════════════════════════════════════════════════════════════
# 8. PANEL TESTLERİ (Panel-Specific Threshold)
# ══════════════════════════════════════════════════════════════════

@zaman_olc
def panel_testleri(pipeline, master_ids, X_cols, freq_maps, rapor):
    """Panel testleri — her panel için ayrı threshold optimizasyonu."""
    panel_dosya = {
        'KANSER': VERI_DIR / 'YARISMA_TRAIN_KANSER.csv',
        'PAH': VERI_DIR / 'YARISMA_TRAIN_PAH.csv',
        'CFTR': VERI_DIR / 'YARISMA_TRAIN_CFTR.csv'
    }
    panel_thresholds = {}

    print("\n" + "=" * 60)
    log("PANEL TEST SONUCLARI", "STEP")
    print("=" * 60)
    rapor += "\n--- PANEL TEST SONUCLARI ---\n"

    for panel_adi, dosya in panel_dosya.items():
        if not dosya.exists():
            log(f"{dosya} bulunamadi, atlaniyor.", "WARN")
            continue
        df_p = pd.read_csv(dosya)
        toplam = len(df_p)

        # Tüm veri ile test
        df_all_fe, _, _ = feature_engineering(df_p.copy(), freq_maps=freq_maps)
        y_all = df_all_fe['Label'].values
        X_all = df_all_fe.drop(columns=['Label']).reindex(columns=X_cols, fill_value=0)
        probs_all, _ = ensemble_tahmin(pipeline, X_all, use_meta=False)

        # Panel-specific threshold (F1+MCC kompozit)
        th_panel = optimal_threshold_bul(y_all, probs_all, metrik='f1_mcc')
        panel_thresholds[panel_adi] = th_panel
        preds_all = (probs_all >= th_panel).astype(int)

        f1_all = f1_score(y_all, preds_all, zero_division=0)
        mcc_all = matthews_corrcoef(y_all, preds_all)
        pr_auc_all = average_precision_score(y_all, probs_all)
        prec_all = precision_score(y_all, preds_all, zero_division=0)
        rec_all = recall_score(y_all, preds_all, zero_division=0)
        cm_all = confusion_matrix(y_all, preds_all)
        if cm_all.shape == (2, 2):
            tn, fp, fn, tp = cm_all.ravel()
        else:
            tn = fp = fn = tp = 0

        panel_sonuc = (
            f"\n[{panel_adi} Paneli] (Toplam: {toplam} varyant)\n"
            f"  Threshold:  {th_panel:.2f}\n"
            f"  F1-Skoru:   {f1_all:.4f}\n"
            f"  PR-AUC:     {pr_auc_all:.4f}\n"
            f"  MCC:        {mcc_all:.4f}\n"
            f"  Precision:  {prec_all:.4f}\n"
            f"  Recall:     {rec_all:.4f}\n"
            f"  Karisiklik Matrisi -> TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}\n"
        )

        # Overlap çıkarılmış bağımsız test
        if 'Variant_ID' in df_p.columns and master_ids is not None:
            overlap = df_p['Variant_ID'].isin(master_ids)
            df_indep = df_p[~overlap].copy()
            if len(df_indep) >= 10:
                df_indep_fe, _, _ = feature_engineering(df_indep, freq_maps=freq_maps)
                y_ind = df_indep_fe['Label'].values
                X_ind = df_indep_fe.drop(columns=['Label']).reindex(columns=X_cols, fill_value=0)
                pr_ind, _ = ensemble_tahmin(pipeline, X_ind, use_meta=False)
                th_ind = optimal_threshold_bul(y_ind, pr_ind, metrik='f1_mcc')
                pd_ind = (pr_ind >= th_ind).astype(int)
                panel_sonuc += (
                    f"  [Bagimsiz Test] ({len(df_indep)}/{toplam}, Overlap cikarildi: {overlap.sum()})\n"
                    f"    F1: {f1_score(y_ind, pd_ind, zero_division=0):.4f} | "
                    f"MCC: {matthews_corrcoef(y_ind, pd_ind):.4f} | "
                    f"PR-AUC: {average_precision_score(y_ind, pr_ind):.4f}\n"
                )

        # Klinik Stres Testi bilgisi
        test_dag = TEST_DAGILIM.get(panel_adi, None)
        if test_dag:
            test_p_ratio = test_dag['pathogenic'] / (test_dag['pathogenic'] + test_dag['benign'])
            panel_sonuc += (
                f"  [Klinik Stres Testi Bilgisi] Test dagilimi: "
                f"{test_dag['pathogenic']}P + {test_dag['benign']}B "
                f"(Patojenik orani: %{test_p_ratio*100:.1f})\n"
            )

        print(panel_sonuc)
        rapor += panel_sonuc

    # Panel threshold'ları kaydet
    pipeline['panel_thresholds'] = panel_thresholds
    joblib.dump(pipeline, SONUC_DIR / 'kgm_med39_pipeline.joblib')
    log(f"Panel threshold'lari kaydedildi: {panel_thresholds}", "OK")

    return rapor


# ══════════════════════════════════════════════════════════════════
# 9. SHAP ANALİZİ
# ══════════════════════════════════════════════════════════════════

@zaman_olc
def shap_analizi(pipeline, X, y):
    """SHAP açıklanabilirlik analizi."""
    log("SHAP analizi basliyor...", "STEP")
    X_sc = pipeline['scaler'].transform(pipeline['imputer'].transform(X))

    # En yüksek ağırlıklı tree-based model
    tree_models = {k: v for k, v in pipeline['models'].items()
                   if k in ('xgb', 'lgbm', 'catboost', 'et')}
    ana_model_adi = max(tree_models,
                       key=lambda k: pipeline['model_weights'].get(k, 0))
    ana_model = pipeline['models'][ana_model_adi]
    log(f"SHAP: '{ana_model_adi}' (agirlik: {pipeline['model_weights'][ana_model_adi]:.3f})", "INFO")

    explainer = shap.TreeExplainer(ana_model)
    shap_values = explainer.shap_values(X_sc)

    # Summary Plot
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])
    ax_shap = fig.add_subplot(gs[0])
    plt.sca(ax_shap)
    shap.summary_plot(shap_values, X_sc, feature_names=X.columns,
                      show=False, max_display=25)
    ax_shap.set_title(
        f'KGM_MED39 v4.0 - SHAP Ozellik Onem Analizi ({ana_model_adi.upper()})',
        fontsize=14, fontweight='bold', pad=15)

    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')
    bilgi = (
        "SHAP GRAFIGI YORUMLAMA REHBERI\n"
        "=" * 32 + "\n\n"
        "1. ONEM SIRASI (Yukaridan Asagiya):\n"
        "   En ustte yer alan ozellik,\n"
        "   modelin en cok guvendigi\n"
        "   kolondur.\n\n"
        "2. KARAR YONU (Sag ve Sol):\n"
        "   Sag = 'Patojenik (1)'\n"
        "   Sol = 'Benign (0)'\n\n"
        "3. RENK (Kirmizi/Mavi):\n"
        "   Kirmizi = Yuksek deger\n"
        "   Mavi = Dusuk deger\n\n"
        "4. DAGILIM GENISLIGI:\n"
        "   Yatay yayilim = Etki gucu\n\n"
        "=" * 32 + "\n"
        "v4.0 Stacking Ensemble\n"
        f" Ana model: {ana_model_adi.upper()}\n"
        " + Meta-Learner (LR)\n"
    )
    ax_text.text(0.0, 0.5, bilgi, fontsize=10, va='center', ha='left',
                 fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=1", fc="#f0f0f0",
                           ec="#888888", alpha=0.95))
    plt.tight_layout()
    plt.savefig(SONUC_DIR / 'shap_aciklanabilirlik_grafigi.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    log("SHAP summary plot kaydedildi", "OK")

    # Feature Importance Bar
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sc, feature_names=X.columns,
                      plot_type="bar", show=False, max_display=30)
    plt.title('KGM_MED39 v4.0 - Ozellik Onem Siralamasi (Mean |SHAP|)',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SONUC_DIR / 'shap_feature_importance.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    log("SHAP feature importance kaydedildi", "OK")

    # Dependence Plots (Top 6)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-6:][::-1]
    fig3, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, fi in enumerate(top_idx):
        plt.sca(axes[i])
        shap.dependence_plot(fi, shap_values, X_sc,
                             feature_names=X.columns,
                             show=False, ax=axes[i])
        axes[i].set_title(f'{X.columns[fi]}', fontsize=11, fontweight='bold')
    fig3.suptitle('KGM_MED39 v4.0 - SHAP Dependence Plots (Top 6)',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(SONUC_DIR / 'shap_dependence_plots.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    log("SHAP dependence plots kaydedildi", "OK")
    return shap_values


# ══════════════════════════════════════════════════════════════════
# 10. RAPORLAMA VE GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════════

def rapor_grafikleri(sonuclar, y):
    """Fold karşılaştırma, PR eğrisi, karmaşıklık matrisi grafikleri."""

    # Fold Karşılaştırma
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, scores, title, color in zip(
        axes,
        [[fd['f1'] for fd in sonuclar['fold_detaylari']],
         [fd['pr_auc'] for fd in sonuclar['fold_detaylari']],
         [fd['mcc'] for fd in sonuclar['fold_detaylari']]],
        ['F1-Score', 'PR-AUC', 'MCC'],
        ['#2196F3', '#4CAF50', '#FF9800']
    ):
        ax.bar(range(1, len(scores)+1), scores, color=color, alpha=0.7,
               edgecolor='black', linewidth=0.5)
        ax.axhline(y=np.mean(scores), color='red', linestyle='--',
                   linewidth=1.5,
                   label=f'Ort: {np.mean(scores):.4f} +/- {np.std(scores):.4f}')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Fold')
        ax.set_ylabel(title)
        ax.legend(fontsize=9)
        ax.set_ylim(max(0, min(scores) - 0.05), min(max(scores) + 0.05, 1.0))
    fig.suptitle('KGM_MED39 v4.0 - Capraz Dogrulama Fold Karsilastirmasi',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(SONUC_DIR / 'fold_karsilastirma.png', dpi=300, bbox_inches='tight')
    plt.close()
    log("Fold karsilastirma grafigi kaydedildi", "OK")

    # PR Eğrisi
    prec_arr, rec_arr, _ = precision_recall_curve(y, sonuclar['oof_probs'])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_arr, prec_arr, 'b-', linewidth=2,
            label=f'PR-AUC = {sonuclar["ort_pr_auc"]:.4f}')
    ax.fill_between(rec_arr, prec_arr, alpha=0.1, color='blue')
    ax.set_xlabel('Recall (Duyarlilik)', fontsize=12)
    ax.set_ylabel('Precision (Kesinlik)', fontsize=12)
    ax.set_title('KGM_MED39 v4.0 - Precision-Recall Egrisi (OOF)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SONUC_DIR / 'precision_recall_egrisi.png', dpi=300, bbox_inches='tight')
    plt.close()
    log("PR egrisi kaydedildi", "OK")

    # Karmaşıklık Matrisi
    th = sonuclar['global_threshold']
    preds = (sonuclar['oof_probs'] >= th).astype(int)
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign (0)', 'Pathogenic (1)'], fontsize=11)
    ax.set_yticklabels(['Benign (0)', 'Pathogenic (1)'], fontsize=11)
    ax.set_xlabel('Tahmin', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gercek', fontsize=13, fontweight='bold')
    ax.set_title(f'KGM_MED39 v4.0 - Karisiklik Matrisi (Threshold: {th:.2f})',
                 fontsize=13, fontweight='bold')
    for i in range(2):
        for j in range(2):
            clr = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    fontsize=18, fontweight='bold', color=clr)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(SONUC_DIR / 'karisiklik_matrisi.png', dpi=300, bbox_inches='tight')
    plt.close()
    log("Karisiklik matrisi kaydedildi", "OK")


# ══════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("\n" + "=" * 60)
    print("  KGM_MED39 v4.0 - Klinik Genomik Varyant Tahmin Sistemi")
    print("  TEKNOFEST 2026 - Saglikta Yapay Zeka Yarismasi")
    print("  Championship Edition: Stacking + Optuna + Stres Testi")
    print("=" * 60 + "\n")

    rapor = "KGM_MED39 - KLINIK YAPAY ZEKA TEST RAPORU (v4.0 Championship)\n"
    rapor += "=" * 60 + "\n"
    rapor += f"Tarih: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # ── 1. VERİ YÜKLEME ──
    print("=" * 60)
    log("ADIM 1: Veri Yukleme", "STEP")
    print("=" * 60)
    df = veri_yukle(VERI_DIR / 'YARISMA_TRAIN_MASTER.csv')
    master_ids = df['Variant_ID'].copy() if 'Variant_ID' in df.columns else None

    # ── 2. FEATURE ENGINEERING ──
    print("\n" + "=" * 60)
    log("ADIM 2: Gelismis Feature Engineering", "STEP")
    print("=" * 60)
    df_fe, _, freq_maps = feature_engineering(df, freq_maps=None)
    log(f"FE sonrasi kolon sayisi: {len(df_fe.columns)}", "OK")

    # ── 3. VERİ HAZIRLAMA ──
    print("\n" + "=" * 60)
    log("ADIM 3: Veri Hazirlama", "STEP")
    print("=" * 60)
    X, y = veri_hazirla(df_fe)
    log(f"Son veri boyutu: {X.shape}", "OK")
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    log(f"scale_pos_weight: {spw:.4f}", "INFO")

    # Klinik Stres Testi bilgisi
    test_info = TEST_DAGILIM['MASTER']
    log(f"[SARTNAME] Test dagilimi: {test_info['pathogenic']}P + {test_info['benign']}B "
        f"(Patojenik: %{test_info['pathogenic']/(test_info['pathogenic']+test_info['benign'])*100:.1f})", "WARN")

    # ── 4. OPTUNA OPTİMİZASYON ──
    print("\n" + "=" * 60)
    log("ADIM 4: Optuna Hiperparametre Optimizasyonu (Tum Modeller)", "STEP")
    print("=" * 60)
    best_params = optuna_optimize(X, y, spw)
    if best_params:
        rapor += "--- OPTIMUM HIPERPARAMETRELER ---\n"
        for mn, params in best_params.items():
            rapor += f"\n[{mn.upper()}]\n"
            for k, v in params.items():
                rapor += f"  {k}: {v}\n"
        rapor += "\n"

    # ── 5. MODEL OLUŞTURMA ──
    print("\n" + "=" * 60)
    log("ADIM 5: Stacking Ensemble Model Olusturma", "STEP")
    print("=" * 60)
    models = modelleri_olustur(spw, best_params)
    log(f"Aktif modeller: {list(models.keys())}", "OK")

    # ── 6. ÇAPRAZ DOĞRULAMA (Stacking + Stres Testi) ──
    print("\n" + "=" * 60)
    log("ADIM 6: Stacking Capraz Dogrulama + Klinik Stres Testi", "STEP")
    print("=" * 60)
    sonuclar = capraz_dogrulama_stacking(X, y, models)

    print("\n" + "-" * 50)
    log(f"Ortalama F1:        {sonuclar['ort_f1']:.4f} +/- {sonuclar['std_f1']:.4f}", "OK")
    log(f"Ortalama PR-AUC:    {sonuclar['ort_pr_auc']:.4f} +/- {sonuclar['std_pr_auc']:.4f}", "OK")
    log(f"Ortalama MCC:       {sonuclar['ort_mcc']:.4f} +/- {sonuclar['std_mcc']:.4f}", "OK")
    log(f"Ortalama Precision: {sonuclar['ort_prec']:.4f}", "OK")
    log(f"Ortalama Recall:    {sonuclar['ort_rec']:.4f}", "OK")
    log(f"Global Threshold:   {sonuclar['global_threshold']:.2f}", "OK")
    log(f"Meta-Learner F1:    {sonuclar['meta_f1']:.4f}", "STAR")
    log(f"Meta-Learner MCC:   {sonuclar['meta_mcc']:.4f}", "STAR")
    if sonuclar['stress_f1']:
        log(f"Stres Testi Ort F1: {sonuclar['stress_f1']:.4f}", "WARN")
        log(f"Stres Testi Ort MCC:{sonuclar['stress_mcc']:.4f}", "WARN")
    print("-" * 50)

    rapor += "--- ANA MODEL (MASTER) STACKING ENSEMBLE SONUCLARI ---\n"
    rapor += "-" * 50 + "\n"
    rapor += f"Aktif Modeller: {', '.join(models.keys())}\n"
    rapor += f"CV: {N_SPLITS}-Fold x {N_REPEATS}-Repeat = {N_SPLITS*N_REPEATS} Fold\n"
    rapor += f"Meta-Learner: LogisticRegression (Stacking Level-2)\n\n"
    rapor += f"Ortalama F1:        {sonuclar['ort_f1']:.4f} +/- {sonuclar['std_f1']:.4f}\n"
    rapor += f"Ortalama PR-AUC:    {sonuclar['ort_pr_auc']:.4f} +/- {sonuclar['std_pr_auc']:.4f}\n"
    rapor += f"Ortalama MCC:       {sonuclar['ort_mcc']:.4f} +/- {sonuclar['std_mcc']:.4f}\n"
    rapor += f"Ortalama Precision: {sonuclar['ort_prec']:.4f}\n"
    rapor += f"Ortalama Recall:    {sonuclar['ort_rec']:.4f}\n"
    rapor += f"Global Threshold:   {sonuclar['global_threshold']:.2f}\n"
    rapor += f"Meta-Learner F1:    {sonuclar['meta_f1']:.4f}\n"
    rapor += f"Meta-Learner MCC:   {sonuclar['meta_mcc']:.4f}\n"
    if sonuclar['stress_f1']:
        rapor += f"\n--- KLINIK STRES TESTI SIMULASYONU ---\n"
        rapor += f"Stres Testi Ort F1:  {sonuclar['stress_f1']:.4f}\n"
        rapor += f"Stres Testi Ort MCC: {sonuclar['stress_mcc']:.4f}\n"
    rapor += "-" * 50 + "\n\n"
    rapor += "Fold Detaylari:\n"
    for fd in sonuclar['fold_detaylari']:
        rapor += (f"  Fold {fd['fold']:2d}: F1={fd['f1']:.4f} | PR-AUC={fd['pr_auc']:.4f} | "
                  f"MCC={fd['mcc']:.4f} | Thresh={fd['threshold']:.2f}\n")
    rapor += "\n"

    # ── 7. FİNAL MODEL ──
    print("\n" + "=" * 60)
    log("ADIM 7: Final Model Egitimi (Stacking Pipeline)", "STEP")
    print("=" * 60)
    pipeline = final_model_egit(X, y, models, sonuclar)
    rapor += "--- MODEL AGIRLIKLARI ---\n"
    for n, w in pipeline['model_weights'].items():
        rapor += f"  {n}: {w:.4f}\n"
    rapor += "\n"

    # Freq maps'i de kaydet
    joblib.dump(freq_maps, SONUC_DIR / 'freq_maps.joblib')

    # ── 8. PANEL TESTLERİ ──
    print("\n" + "=" * 60)
    log("ADIM 8: Panel Testleri (Panel-Specific Threshold)", "STEP")
    print("=" * 60)
    rapor = panel_testleri(pipeline, master_ids, X.columns, freq_maps, rapor)

    # ── 9. RAPORLAMA ──
    print("\n" + "=" * 60)
    log("ADIM 9: Raporlama ve Gorsellestirme", "STEP")
    print("=" * 60)
    rapor_grafikleri(sonuclar, y)
    shap_analizi(pipeline, X, y)

    # Klinik Stres Testi bilgisi
    rapor += "\n--- KLINIK STRES TESTI BILGISI ---\n"
    rapor += "(Sartname Sayfa 8'den alinan test veri dagilimi)\n"
    for panel, dag in TEST_DAGILIM.items():
        oran = dag['pathogenic'] / (dag['pathogenic'] + dag['benign']) * 100
        rapor += f"  {panel}: {dag['pathogenic']}P + {dag['benign']}B (Patojenik: %{oran:.1f})\n"
    rapor += "\n"

    rapor += "=" * 60 + "\n"
    rapor += f"Toplam Calisma Suresi: {time.time() - t0:.1f} saniye\n"
    rapor += f"Feature Engineering: {len(X.columns)} ozellik\n"
    rapor += f"Ensemble: {', '.join(models.keys())} + Meta-Learner\n"
    rapor += f"Versiyon: v4.0 Championship Edition\n"
    rapor += "=" * 60 + "\n"

    with open(SONUC_DIR / 'test_raporu.txt', 'w', encoding='utf-8') as f:
        f.write(rapor)

    # ── ÖZET ──
    print("\n" + "=" * 60)
    print("  ISLEM TAMAMLANDI")
    print("=" * 60)
    log(f"Toplam sure: {time.time() - t0:.1f} saniye", "OK")
    log(f"Sonuclar: {SONUC_DIR}", "OK")
    print("\nOlusturulan dosyalar:")
    for f in sorted(SONUC_DIR.glob('*')):
        sz = f.stat().st_size / 1024
        print(f"   - {f.name} ({sz:.1f} KB)")
    print()


if __name__ == '__main__':
    main()