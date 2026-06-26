#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KGM_MED39 — Ortak Feature Engineering Modülü

Eğitim ve tahmin scriptlerinde aynı FE fonksiyonu kullanılır.
Bu sayede eğitim/tahmin arasındaki tutarsızlık riski ortadan kalkar (DRY prensibi).

Özellik Kategorileri (Şartname Sayfa 9):
  AL_ : Frekans/Popülasyon verileri
  EK_ : Evrimsel Korunmuşluk skorları
  CAT_: Kategorik Meta-Veri
  AA_ : Amino Asit Değişimi
  FE_ : Mühendislik ile üretilen türetilmiş özellikler
"""

import warnings
import numpy as np
import pandas as pd
from .config import (
    AA_HIDROFOBIK, AA_POLAR, AA_POZITIF, AA_NEGATIF, AA_KUCUK,
    EK_PAIRS
)

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


def feature_engineering(df, freq_maps=None):
    """
    Şartname bilgisine dayalı gelişmiş feature engineering.

    Args:
        df: Ham DataFrame (Label kolonu dahil olabilir)
        freq_maps: Eğitim sırasında oluşturulan frekans haritaları.
                   None ise fit mode (eğitim), değilse transform mode (tahmin).

    Returns:
        df_fe: İşlenmiş DataFrame
        variant_ids: Variant_ID serisi (veya None)
        freq_maps: Frekans haritaları (fit modda oluşturulur)
    """
    df = df.copy()
    fit_mode = freq_maps is None

    # ─── Variant_ID sakla ve düşür ───
    variant_ids = None
    if 'Variant_ID' in df.columns:
        variant_ids = df['Variant_ID'].copy()
        df = df.drop(columns=['Variant_ID'])

    # ─── Label ayır ───
    label = None
    if 'Label' in df.columns:
        label = df['Label'].copy()
        df = df.drop(columns=['Label'])

    # ─── Kolon grupları ───
    al_cols = sorted([c for c in df.columns if c.startswith('AL_')])
    ek_cols = sorted([c for c in df.columns if c.startswith('EK_')])
    cat_cols = sorted([c for c in df.columns if c.startswith('CAT_')])
    aa_cols = sorted([c for c in df.columns if c.startswith('AA_')])

    al_num = [c for c in al_cols if df[c].dtype != 'object']
    ek_num = [c for c in ek_cols if df[c].dtype != 'object']

    # ═══════════════════════════════════════════════════
    # 1. AL (Frekans/Popülasyon) İstatistikleri
    # ═══════════════════════════════════════════════════
    if al_num:
        al_data = df[al_num]
        df['FE_AL_mean'] = al_data.mean(axis=1)
        df['FE_AL_std'] = al_data.std(axis=1)
        df['FE_AL_min'] = al_data.min(axis=1)
        df['FE_AL_max'] = al_data.max(axis=1)
        df['FE_AL_median'] = al_data.median(axis=1)
        df['FE_AL_range'] = df['FE_AL_max'] - df['FE_AL_min']
        df['FE_AL_q25'] = al_data.quantile(0.25, axis=1)
        df['FE_AL_q75'] = al_data.quantile(0.75, axis=1)
        df['FE_AL_iqr'] = df['FE_AL_q75'] - df['FE_AL_q25']
        df['FE_AL_skew'] = al_data.skew(axis=1)
        df['FE_AL_kurtosis'] = al_data.kurtosis(axis=1)
        df['FE_AL_nonzero'] = (al_data != 0).sum(axis=1)
        df['FE_AL_ones'] = (al_data == 1.0).sum(axis=1)
        df['FE_AL_near_zero'] = (al_data.abs() < 1e-5).sum(axis=1)
        # Domain bilgisi: Yüksek frekans = Benign olma ihtimali yüksek
        df['FE_AL_high_freq_count'] = (al_data > 0.01).sum(axis=1)
        df['FE_AL_rare_count'] = ((al_data > 0) & (al_data < 0.001)).sum(axis=1)
        # Ek: Frekans entropisi (bilgi yoğunluğu)
        al_pos = al_data.clip(lower=1e-10)
        df['FE_AL_entropy'] = -(al_pos * np.log2(al_pos + 1e-10)).sum(axis=1)
        # Ek: Geometric mean (nadir varyant tespiti)
        df['FE_AL_log_mean'] = np.log1p(al_data.clip(lower=0)).mean(axis=1)
        # Coefficient of variation
        df['FE_AL_cv'] = df['FE_AL_std'] / (df['FE_AL_mean'].abs() + 1e-10)

    # ═══════════════════════════════════════════════════
    # 2. EK (Evrimsel Korunmuşluk) İstatistikleri
    # ═══════════════════════════════════════════════════
    if ek_num:
        ek_data = df[ek_num]
        df['FE_EK_mean'] = ek_data.mean(axis=1)
        df['FE_EK_std'] = ek_data.std(axis=1)
        df['FE_EK_min'] = ek_data.min(axis=1)
        df['FE_EK_max'] = ek_data.max(axis=1)
        df['FE_EK_range'] = df['FE_EK_max'] - df['FE_EK_min']
        df['FE_EK_sum'] = ek_data.sum(axis=1)
        df['FE_EK_median'] = ek_data.median(axis=1)
        # Domain bilgisi: Evrimsel korunmuşluk yüksek = Patojenik olma ihtimali yüksek
        df['FE_EK_high_cons'] = (ek_data > 0.8).sum(axis=1)
        df['FE_EK_low_cons'] = (ek_data < 0.2).sum(axis=1)
        # Ek: Korunmuşluk skoru uyumu (concordance)
        df['FE_EK_cv'] = df['FE_EK_std'] / (df['FE_EK_mean'].abs() + 1e-10)
        # Ek: Pozitif/negatif korunmuşluk oranı
        df['FE_EK_pos_ratio'] = (ek_data > 0).sum(axis=1) / max(len(ek_num), 1)

    # ═══════════════════════════════════════════════════
    # 3. Eksik Veri Pattern (Bilgilendirici)
    # ═══════════════════════════════════════════════════
    df['FE_missing_total'] = df.isnull().sum(axis=1)
    df['FE_missing_ratio'] = df['FE_missing_total'] / max(len(df.columns), 1)
    if al_cols:
        df['FE_AL_missing'] = df[al_cols].isnull().sum(axis=1)
        df['FE_AL_missing_ratio'] = df['FE_AL_missing'] / max(len(al_cols), 1)
    if ek_cols:
        df['FE_EK_missing'] = df[ek_cols].isnull().sum(axis=1)
        df['FE_EK_missing_ratio'] = df['FE_EK_missing'] / max(len(ek_cols), 1)

    # ═══════════════════════════════════════════════════
    # 4. Özellik Etkileşimleri (SHAP yönlendirmeli)
    # ═══════════════════════════════════════════════════
    for c1, c2 in EK_PAIRS:
        if c1 in df.columns and c2 in df.columns:
            df[f'FE_{c1}x{c2}'] = df[c1] * df[c2]
            df[f'FE_{c1}-{c2}'] = df[c1] - df[c2]
            df[f'FE_{c1}d{c2}'] = df[c1] / (df[c2].abs() + 1e-10)

    # EK-AL çapraz etkileşimler (domain bilgisi:
    # frekans düşük + korunmuşluk yüksek = muhtemelen patojenik)
    if 'EK_7' in df.columns and 'AL_327' in df.columns:
        df['FE_EK7xAL327'] = df['EK_7'] * df['AL_327']
    if 'FE_EK_mean' in df.columns and 'FE_AL_mean' in df.columns:
        df['FE_EKmean_x_ALmean'] = df['FE_EK_mean'] * df['FE_AL_mean']
        df['FE_EKmean_d_ALmean'] = df['FE_EK_mean'] / (df['FE_AL_mean'].abs() + 1e-10)

    # ═══════════════════════════════════════════════════
    # 5. Log dönüşümleri (EK skorları — dağılım normalleştirme)
    # ═══════════════════════════════════════════════════
    for c in ek_num:
        if c in df.columns:
            df[f'FE_{c}_log'] = np.log1p(df[c].clip(lower=0))

    # ═══════════════════════════════════════════════════
    # 6. Amino Asit Özellikleri (Biyokimyasal bilgi)
    # ═══════════════════════════════════════════════════
    if 'AA_1' in df.columns and 'AA_2' in df.columns:
        df['FE_AA_same'] = (df['AA_1'] == df['AA_2']).astype(int)

        df['FE_AA1_hydro'] = df['AA_1'].apply(
            lambda x: 1 if str(x) in AA_HIDROFOBIK else 0)
        df['FE_AA2_hydro'] = df['AA_2'].apply(
            lambda x: 1 if str(x) in AA_HIDROFOBIK else 0)
        df['FE_AA_hydro_change'] = (
            df['FE_AA1_hydro'] != df['FE_AA2_hydro']).astype(int)

        def yuk(aa):
            aa = str(aa)
            if aa in AA_POZITIF:
                return 1
            elif aa in AA_NEGATIF:
                return -1
            return 0

        df['FE_AA1_charge'] = df['AA_1'].apply(yuk)
        df['FE_AA2_charge'] = df['AA_2'].apply(yuk)
        df['FE_AA_charge_change'] = (
            df['FE_AA1_charge'] != df['FE_AA2_charge']).astype(int)
        df['FE_AA_charge_diff'] = (
            df['FE_AA1_charge'] - df['FE_AA2_charge']).abs()

        df['FE_AA1_small'] = df['AA_1'].apply(
            lambda x: 1 if str(x) in AA_KUCUK else 0)
        df['FE_AA2_small'] = df['AA_2'].apply(
            lambda x: 1 if str(x) in AA_KUCUK else 0)
        df['FE_AA_size_change'] = (
            df['FE_AA1_small'] != df['FE_AA2_small']).astype(int)

        df['FE_AA1_polar'] = df['AA_1'].apply(
            lambda x: 1 if str(x) in AA_POLAR else 0)
        df['FE_AA2_polar'] = df['AA_2'].apply(
            lambda x: 1 if str(x) in AA_POLAR else 0)
        df['FE_AA_polarity_change'] = (
            df['FE_AA1_polar'] != df['FE_AA2_polar']).astype(int)

        # Grantham skoru tahmini (radikal değişim skoru)
        df['FE_AA_radical'] = (
            df['FE_AA_hydro_change']
            + df['FE_AA_charge_change']
            + df['FE_AA_size_change']
            + df['FE_AA_polarity_change']
        )

        # Etkileşim: Radikal AA değişimi + yüksek evrimsel korunmuşluk
        if 'FE_EK_mean' in df.columns:
            df['FE_radical_x_conservation'] = (
                df['FE_AA_radical'] * df['FE_EK_mean'])

    # ═══════════════════════════════════════════════════
    # 7. Kategorik & AA Frekans Encoding
    # ═══════════════════════════════════════════════════
    if fit_mode:
        freq_maps = {}

    for c in cat_cols:
        if c in df.columns:
            if fit_mode:
                fmap = df[c].value_counts(normalize=True).to_dict()
                freq_maps[c] = fmap
            else:
                fmap = freq_maps.get(c, {})
            df[f'FE_{c}_freq'] = df[c].map(fmap).fillna(0)

    for c in aa_cols:
        if c in df.columns:
            if fit_mode:
                fmap = df[c].astype(str).value_counts(normalize=True).to_dict()
                freq_maps[c] = fmap
            else:
                fmap = freq_maps.get(c, {})
            df[f'FE_{c}_freq'] = df[c].astype(str).map(fmap).fillna(0)

    # ═══════════════════════════════════════════════════
    # 8. One-hot encoding (kalan kategorik sütunlar)
    # ═══════════════════════════════════════════════════
    obj_cols = [c for c in df.columns if df[c].dtype == 'object']
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # ─── Label geri ekle ───
    if label is not None:
        df['Label'] = label.values

    return df, variant_ids, freq_maps
