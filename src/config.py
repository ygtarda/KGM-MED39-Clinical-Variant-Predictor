#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KGM_MED39 — Merkezi Konfigürasyon Modülü
Tüm sabitler, yollar ve şartname parametreleri burada tanımlanır.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════
# YOLLAR
# ══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent
VERI_DIR = BASE_DIR / 'veriler'
SONUC_DIR = BASE_DIR / 'sonuclar'
SONUC_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════
# MODEL PARAMETRELERİ
# ══════════════════════════════════════════════════════════════
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 2
MISSING_THRESH_RATIO = 0.70
OPTUNA_N_TRIALS = 60
OPTUNA_TIMEOUT = 600

# ══════════════════════════════════════════════════════════════
# ŞARTNAME TEST DAĞILIMLARI (Klinik Stres Testi)
# Sayfa 8'den alınmıştır.
# Eğitim: Patojenik baskın, Test: Benign baskın
# ══════════════════════════════════════════════════════════════
TEST_DAGILIM = {
    'MASTER':  {'pathogenic': 500,  'benign': 3000},
    'KANSER':  {'pathogenic': 100,  'benign': 500},
    'PAH':     {'pathogenic': 100,  'benign': 250},
    'CFTR':    {'pathogenic': 20,   'benign': 100},
}

# ══════════════════════════════════════════════════════════════
# VERİ DOSYALARI
# ══════════════════════════════════════════════════════════════
PANEL_DOSYALARI = {
    'MASTER': VERI_DIR / 'YARISMA_TRAIN_MASTER.csv',
    'KANSER': VERI_DIR / 'YARISMA_TRAIN_KANSER.csv',
    'PAH':    VERI_DIR / 'YARISMA_TRAIN_PAH.csv',
    'CFTR':   VERI_DIR / 'YARISMA_TRAIN_CFTR.csv',
}

# ══════════════════════════════════════════════════════════════
# AMİNO ASİT ÖZELLİKLERİ (Domain Bilgisi)
# ══════════════════════════════════════════════════════════════
AA_HIDROFOBIK = set('AVILMFWP')
AA_POLAR = set('STYHCNQDE')
AA_POZITIF = set('KRH')
AA_NEGATIF = set('DE')
AA_KUCUK = set('GAVSTC')

# Grantham Matrix (Amino asit değişim şiddeti skorları)
# Radikal değişimler patojenik olma eğilimindedir
GRANTHAM_GROUPS = {
    'nonpolar': set('GAVLIPFWM'),
    'polar_uncharged': set('STCYNQ'),
    'positive': set('KRH'),
    'negative': set('DE'),
}

# ══════════════════════════════════════════════════════════════
# EK ÇİFTLERİ (SHAP yönlendirmeli etkileşim özellikleri)
# ══════════════════════════════════════════════════════════════
EK_PAIRS = [
    ('EK_7', 'EK_9'), ('EK_7', 'EK_4'), ('EK_7', 'EK_2'),
    ('EK_9', 'EK_4'), ('EK_9', 'EK_2'), ('EK_4', 'EK_2'),
    ('EK_7', 'EK_6'), ('EK_7', 'EK_3'), ('EK_7', 'EK_8'),
    ('EK_9', 'EK_6'), ('EK_4', 'EK_3'), ('EK_2', 'EK_6'),
]
