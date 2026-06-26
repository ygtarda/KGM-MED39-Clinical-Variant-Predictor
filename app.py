#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║  KGM_MED39 - Klinik Genomik Varyant Tahmin Dashboard             ║
║  TEKNOFEST 2026 - Sağlıkta Yapay Zeka Yarışması                  ║
║  Streamlit Professional Dashboard v4.0                            ║
╚══════════════════════════════════════════════════════════════════╝

Kullanım:
    streamlit run app.py
"""

# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Proje kök dizini
BASE_DIR = Path(__file__).resolve().parent
SONUC_DIR = BASE_DIR / 'sonuclar'
sys.path.insert(0, str(BASE_DIR))

from src.feature_engineering import feature_engineering
from src.config import TEST_DAGILIM

# ══════════════════════════════════════════════════════════════════
# SAYFA KONFİGÜRASYONU
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="KGM_MED39 | Klinik Varyant Tahmin",
    page_icon="medical_symbol",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# CLINICAL CSS TASARIM
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    /* Ana tema */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Roboto', sans-serif;
        color: #212529;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #dee2e6;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #343a40 !important;
    }

    /* Kartlar */
    .glass-card, .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .metric-value {
        font-size: 2em;
        font-weight: 700;
        color: #0284c7;
        margin: 8px 0;
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.85em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-sub {
        color: #adb5bd;
        font-size: 0.75em;
        margin-top: 4px;
    }

    /* Hero baslik */
    .hero-title {
        font-size: 2.2em;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        margin: 20px 0 10px 0;
    }
    .hero-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1em;
        margin-bottom: 30px;
    }

    /* Durum badge'leri */
    .badge-pathogenic {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
        border: 1px solid #f87171;
    }
    .badge-benign {
        background-color: #d1fae5;
        color: #047857;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
        border: 1px solid #34d399;
    }

    /* Mimari diyagram */
    .arch-box {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 16px;
        text-align: center;
        color: #334155;
        font-weight: 500;
        margin: 8px 4px;
    }

    /* Streamlit overrides */
    h1, h2, h3, h4, h5, h6 { color: #1e293b !important; }
    p, li { color: #334155; }
    
    [data-testid="stFileUploader"] {
        background-color: #f8fafc;
        border-radius: 8px;
        border: 2px dashed #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_pipeline():
    """Pipeline'ı yükle (cache'li)."""
    pipeline_path = SONUC_DIR / 'kgm_med39_pipeline.joblib'
    freq_path = SONUC_DIR / 'freq_maps.joblib'

    if not pipeline_path.exists():
        return None, None

    pipeline = joblib.load(pipeline_path)
    freq_maps = joblib.load(freq_path) if freq_path.exists() else {}
    return pipeline, freq_maps


def predict_batch(pipeline, freq_maps, df):
    """Toplu tahmin üret."""
    df_fe, variant_ids, _ = feature_engineering(df, freq_maps=freq_maps)
    if 'Label' in df_fe.columns:
        df_fe = df_fe.drop(columns=['Label'])

    feature_cols = pipeline['feature_columns']
    X = df_fe.reindex(columns=feature_cols, fill_value=0)

    X_imp = pipeline['imputer'].transform(X)
    X_sc = pipeline['scaler'].transform(X_imp)

    base_probs = []
    weighted_probs = np.zeros(len(X))
    for name, model in pipeline['models'].items():
        w = pipeline['model_weights'].get(name, 1.0 / len(pipeline['models']))
        p = model.predict_proba(X_sc)[:, 1]
        base_probs.append(p)
        weighted_probs += w * p

    # Meta-learner
    if 'meta_learner' in pipeline and pipeline['meta_learner'] is not None:
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
    return probs, preds, threshold, variant_ids


def render_metric_card(label, value, sub="", color_class=""):
    """Glassmorphism metrik kartı render et."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("##  KGM_MED39")
    st.markdown("---")

    page = st.radio(
        "Sayfa Seçin",
        [" Ana Sayfa", " Varyant Tahmin", " Model Performans",
         " Panel Analizi", " Klinik Stres Testi", " YZ Açıklanabilirliği", " Hakkında"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7c7ca0; font-size: 0.75em;'>
        <p>TEKNOFEST 2026</p>
        <p>Sağlıkta Yapay Zeka</p>
        <p style='color: #7c4dff;'>v4.0 Championship</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SAYFA: ANA SAYFA
# ══════════════════════════════════════════════════════════════════

if page == " Ana Sayfa":
    st.markdown('<div class="hero-title"> KGM_MED39</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Klinik Genomik Varyant Patojenite Tahmin Sistemi</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle" style="font-size:0.9em; color:#7c4dff;">TEKNOFEST 2026 — Sağlıkta Yapay Zeka Yarışması — Üniversite ve Üzeri Seviyesi</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Proje özeti
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00d4ff;">📋 Proje Özeti</h3>
            <p>Bu sistem, klinik etkisi bilinmeyen genetik varyantların (VUS - Variant of Uncertain Significance)
            <strong>Patojenik</strong> veya <strong>Benign</strong> olarak sınıflandırılması için geliştirilmiş
            ileri düzey bir makine öğrenmesi pipeline'ıdır.</p>
            <p>ACMG (Amerikan Tıbbi Genetik ve Genomik Koleji) rehberlerine uyumlu etiketleme kullanılmaktadır.</p>
            <ul style="color:#a0a0d0;">
                <li><strong>Stacking Ensemble:</strong> XGBoost + LightGBM + CatBoost + ExtraTrees + Meta-Learner</li>
                <li><strong>Bayesian Optimizasyon:</strong> Optuna ile tüm modeller için otomatik hiperparametre tuning</li>
                <li><strong>Klinik Stres Testi:</strong> Benign-baskın test dağılımı simülasyonu</li>
                <li><strong>SHAP Açıklanabilirlik:</strong> Model kararlarının yorumlanabilir sunumu</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00d4ff;">🏗️ Mimari</h3>
            <div class="arch-box">HAM VERİ (CSV)</div>
            <div style="text-align:center; color:#7c4dff; font-size:1.5em;">↓</div>
            <div class="arch-box">FEATURE ENGINEERING<br><span style="font-size:0.75em;">AL / EK / CAT / AA</span></div>
            <div style="text-align:center; color:#7c4dff; font-size:1.5em;">↓</div>
            <div class="arch-box">OPTUNA OPTİMİZASYON</div>
            <div style="text-align:center; color:#7c4dff; font-size:1.5em;">↓</div>
            <div class="arch-box">STACKING ENSEMBLE<br><span style="font-size:0.75em;">4 Model + Meta-Learner</span></div>
            <div style="text-align:center; color:#7c4dff; font-size:1.5em;">↓</div>
            <div class="arch-box" style="background:rgba(0,212,255,0.15); border-color:rgba(0,212,255,0.4);">
                TAHMİN: Patojenik / Benign
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Hızlı metrikler
    pipeline, freq_maps = load_pipeline()
    if pipeline:
        st.markdown("###  Güncel Model Metrikleri")
        cols = st.columns(5)
        report_path = SONUC_DIR / 'test_raporu.txt'
        if report_path.exists():
            report = report_path.read_text(encoding='utf-8')
            import re
            f1_match = re.search(r'Ortalama F1:\s+([\d.]+)', report)
            mcc_match = re.search(r'Ortalama MCC:\s+([\d.]+)', report)
            prauc_match = re.search(r'Ortalama PR-AUC:\s+([\d.]+)', report)
            prec_match = re.search(r'Ortalama Precision:\s+([\d.]+)', report)
            rec_match = re.search(r'Ortalama Recall:\s+([\d.]+)', report)

            with cols[0]:
                render_metric_card("F1 Score", f1_match.group(1) if f1_match else "—", "Ana Sıralama Metriği")
            with cols[1]:
                render_metric_card("MCC", mcc_match.group(1) if mcc_match else "—", "Matthews Korelasyon")
            with cols[2]:
                render_metric_card("PR-AUC", prauc_match.group(1) if prauc_match else "—", "Precision-Recall AUC")
            with cols[3]:
                render_metric_card("Precision", prec_match.group(1) if prec_match else "—", "Kesinlik")
            with cols[4]:
                render_metric_card("Recall", rec_match.group(1) if rec_match else "—", "Duyarlılık")


# ══════════════════════════════════════════════════════════════════
# SAYFA: VARYANT TAHMİN
# ══════════════════════════════════════════════════════════════════

elif page == " Varyant Tahmin":
    st.markdown('<div class="hero-title" style="font-size:2em;"> Varyant Patojenite Tahmini</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">CSV dosyası yükleyin veya örnek veri ile test edin</div>',
                unsafe_allow_html=True)

    pipeline, freq_maps = load_pipeline()

    if pipeline is None:
        st.error(" Pipeline bulunamadı! Önce `python kgm_med39_predictor.py` çalıştırın.")
        st.stop()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Panel seçimi
    col1, col2 = st.columns([1, 2])
    with col1:
        panel = st.selectbox("Panel Seçin (Opsiyonel)",
                             ["Otomatik", "MASTER", "KANSER", "PAH", "CFTR"])

    # CSV yükleme
    uploaded_file = st.file_uploader(
        "📁 Test CSV Dosyasını Yükleyin",
        type=['csv'],
        help="Yarışma formatındaki CSV dosyasını sürükleyip bırakın"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f" Dosya yüklendi: {df.shape[0]} varyant, {df.shape[1]} kolon")

        with st.expander("📋 Yüklenen Veri Önizleme", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        if st.button(" Tahmin Üret", type="primary", use_container_width=True):
            with st.spinner("🔄 Tahminler hesaplanıyor..."):
                probs, preds, threshold, variant_ids = predict_batch(
                    pipeline, freq_maps, df)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Sonuç özeti
            n_pathogenic = (preds == 1).sum()
            n_benign = (preds == 0).sum()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_card("Toplam", str(len(preds)), "Varyant")
            with col2:
                render_metric_card("Patojenik", str(n_pathogenic),
                                   f"%{n_pathogenic/len(preds)*100:.1f}")
            with col3:
                render_metric_card("Benign", str(n_benign),
                                   f"%{n_benign/len(preds)*100:.1f}")
            with col4:
                render_metric_card("Threshold", f"{threshold:.2f}", "Karar Eşiği")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Dağılım grafiği
            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=probs, nbinsx=50,
                    marker_color='rgba(124,77,255,0.7)',
                    name='Olasılık Dağılımı'
                ))
                fig.add_vline(x=threshold, line_dash="dash",
                              line_color="#00d4ff", line_width=2,
                              annotation_text=f"Threshold: {threshold:.2f}")
                fig.update_layout(
                    title="Patojenik Olasılık Dağılımı",
                    xaxis_title="Olasılık",
                    yaxis_title="Varyant Sayısı",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure(data=[go.Pie(
                    labels=['Benign', 'Patojenik'],
                    values=[n_benign, n_pathogenic],
                    marker_colors=['#00e676', '#ff1744'],
                    hole=0.5,
                    textfont_size=14
                )])
                fig.update_layout(
                    title="Sınıf Dağılımı",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            # Sonuç tablosu
            sonuc_df = pd.DataFrame({
                'Variant_ID': variant_ids if variant_ids is not None else range(len(preds)),
                'Olasılık': np.round(probs, 4),
                'Tahmin': ['Pathogenic' if p == 1 else 'Benign' for p in preds],
            })

            st.markdown("### 📋 Tahmin Sonuçları")
            st.dataframe(sonuc_df, use_container_width=True, height=400)

            # CSV indirme
            csv_data = sonuc_df.to_csv(index=False)
            st.download_button(
                " Sonuçları CSV Olarak İndir",
                csv_data,
                "kgm_med39_tahminler.csv",
                "text/csv",
                use_container_width=True
            )


# ══════════════════════════════════════════════════════════════════
# SAYFA: MODEL PERFORMANS
# ══════════════════════════════════════════════════════════════════

elif page == " Model Performans":
    st.markdown('<div class="hero-title" style="font-size:2em;"> Model Performans Dashboard</div>',
                unsafe_allow_html=True)

    pipeline, _ = load_pipeline()
    report_path = SONUC_DIR / 'test_raporu.txt'

    if not report_path.exists():
        st.warning(" Test raporu bulunamadı. Önce modeli eğitin.")
        st.stop()

    report = report_path.read_text(encoding='utf-8')

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Metrikleri parse et
    import re
    metrics = {}
    for key, pattern in [
        ('f1', r'Ortalama F1:\s+([\d.]+)\s*\+/-\s*([\d.]+)'),
        ('mcc', r'Ortalama MCC:\s+([\d.]+)\s*\+/-\s*([\d.]+)'),
        ('pr_auc', r'Ortalama PR-AUC:\s+([\d.]+)\s*\+/-\s*([\d.]+)'),
        ('precision', r'Ortalama Precision:\s+([\d.]+)'),
        ('recall', r'Ortalama Recall:\s+([\d.]+)'),
        ('threshold', r'Global Threshold:\s+([\d.]+)'),
    ]:
        m = re.search(pattern, report)
        if m:
            metrics[key] = float(m.group(1))
            if m.lastindex >= 2:
                metrics[f'{key}_std'] = float(m.group(2))

    # Metrik kartları
    cols = st.columns(6)
    metric_data = [
        ("F1 Score", metrics.get('f1', 0), f"±{metrics.get('f1_std', 0):.4f}"),
        ("MCC", metrics.get('mcc', 0), f"±{metrics.get('mcc_std', 0):.4f}"),
        ("PR-AUC", metrics.get('pr_auc', 0), f"±{metrics.get('pr_auc_std', 0):.4f}"),
        ("Precision", metrics.get('precision', 0), "Kesinlik"),
        ("Recall", metrics.get('recall', 0), "Duyarlılık"),
        ("Threshold", metrics.get('threshold', 0), "Karar Eşiği"),
    ]
    for col, (label, val, sub) in zip(cols, metric_data):
        with col:
            render_metric_card(label, f"{val:.4f}", sub)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Fold sonuçları grafiği
    fold_data = re.findall(
        r'Fold\s+(\d+):\s+F1=([\d.]+)\s*\|\s*PR-AUC=([\d.]+)\s*\|\s*MCC=([\d.]+)',
        report
    )

    if fold_data:
        folds = [int(f[0]) for f in fold_data]
        f1_scores = [float(f[1]) for f in fold_data]
        prauc_scores = [float(f[2]) for f in fold_data]
        mcc_scores = [float(f[3]) for f in fold_data]

        tab1, tab2 = st.tabs([" Fold Karşılaştırma", "📋 Ham Rapor"])

        with tab1:
            fig = make_subplots(rows=1, cols=3,
                               subplot_titles=('F1 Score', 'PR-AUC', 'MCC'))

            fig.add_trace(go.Bar(x=folds, y=f1_scores, name='F1',
                                marker_color='rgba(33,150,243,0.7)'), row=1, col=1)
            fig.add_hline(y=np.mean(f1_scores), line_dash="dash",
                         line_color="#ff1744", row=1, col=1)

            fig.add_trace(go.Bar(x=folds, y=prauc_scores, name='PR-AUC',
                                marker_color='rgba(76,175,80,0.7)'), row=1, col=2)
            fig.add_hline(y=np.mean(prauc_scores), line_dash="dash",
                         line_color="#ff1744", row=1, col=2)

            fig.add_trace(go.Bar(x=folds, y=mcc_scores, name='MCC',
                                marker_color='rgba(255,152,0,0.7)'), row=1, col=3)
            fig.add_hline(y=np.mean(mcc_scores), line_dash="dash",
                         line_color="#ff1744", row=1, col=3)

            fig.update_layout(
                title="Çapraz Doğrulama Fold Karşılaştırması",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=450, showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.code(report, language='text')

    # Görseller
    st.markdown("### 🖼️ Model Grafikleri")
    img_files = {
        'Karışıklık Matrisi': 'karisiklik_matrisi.png',
        'PR Eğrisi': 'precision_recall_egrisi.png',
        'Fold Karşılaştırma': 'fold_karsilastirma.png',
    }

    cols = st.columns(3)
    for col, (title, fname) in zip(cols, img_files.items()):
        fpath = SONUC_DIR / fname
        if fpath.exists():
            with col:
                st.markdown(f"**{title}**")
                st.image(str(fpath), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# SAYFA: PANEL ANALİZİ
# ══════════════════════════════════════════════════════════════════

elif page == " Panel Analizi":
    st.markdown('<div class="hero-title" style="font-size:2em;"> Panel Bazlı Analiz</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">MASTER / KANSER / PAH / CFTR panelleri ayrı ayrı</div>',
                unsafe_allow_html=True)

    report_path = SONUC_DIR / 'test_raporu.txt'
    if not report_path.exists():
        st.warning(" Test raporu bulunamadı.")
        st.stop()

    report = report_path.read_text(encoding='utf-8')

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Panel sonuçlarını parse et
    import re
    panels = {}
    panel_pattern = re.compile(
        r'\[(\w+) Paneli\].*?Threshold:\s+([\d.]+).*?F1-Skoru:\s+([\d.]+).*?'
        r'PR-AUC:\s+([\d.]+).*?MCC:\s+([\d.]+).*?Precision:\s+([\d.]+).*?'
        r'Recall:\s+([\d.]+).*?TN:\s+(\d+)\s*\|\s*FP:\s+(\d+)\s*\|\s*FN:\s+(\d+)\s*\|\s*TP:\s+(\d+)',
        re.DOTALL
    )
    for m in panel_pattern.finditer(report):
        panels[m.group(1)] = {
            'threshold': float(m.group(2)),
            'f1': float(m.group(3)),
            'pr_auc': float(m.group(4)),
            'mcc': float(m.group(5)),
            'precision': float(m.group(6)),
            'recall': float(m.group(7)),
            'tn': int(m.group(8)),
            'fp': int(m.group(9)),
            'fn': int(m.group(10)),
            'tp': int(m.group(11)),
        }

    if panels:
        # Panel karşılaştırma grafiği
        panel_names = list(panels.keys())
        f1_vals = [panels[p]['f1'] for p in panel_names]
        mcc_vals = [panels[p]['mcc'] for p in panel_names]
        prauc_vals = [panels[p]['pr_auc'] for p in panel_names]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='F1', x=panel_names, y=f1_vals,
                            marker_color='rgba(33,150,243,0.8)'))
        fig.add_trace(go.Bar(name='MCC', x=panel_names, y=mcc_vals,
                            marker_color='rgba(255,152,0,0.8)'))
        fig.add_trace(go.Bar(name='PR-AUC', x=panel_names, y=prauc_vals,
                            marker_color='rgba(76,175,80,0.8)'))
        fig.update_layout(
            title="Panel Bazlı Performans Karşılaştırması",
            barmode='group',
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450, yaxis_range=[0, 1.1]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Panel detay kartları
        for pname, pdata in panels.items():
            with st.expander(f"📋 {pname} Paneli Detayları", expanded=False):
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    render_metric_card("F1", f"{pdata['f1']:.4f}", "")
                with c2:
                    render_metric_card("MCC", f"{pdata['mcc']:.4f}", "")
                with c3:
                    render_metric_card("Precision", f"{pdata['precision']:.4f}", "")
                with c4:
                    render_metric_card("Recall", f"{pdata['recall']:.4f}", "")
                with c5:
                    render_metric_card("Threshold", f"{pdata['threshold']:.2f}", "")

                # Mini confusion matrix
                fig = go.Figure(data=go.Heatmap(
                    z=[[pdata['tn'], pdata['fp']], [pdata['fn'], pdata['tp']]],
                    x=['Benign (Tahmin)', 'Patojenik (Tahmin)'],
                    y=['Benign (Gerçek)', 'Patojenik (Gerçek)'],
                    colorscale='Blues',
                    text=[[str(pdata['tn']), str(pdata['fp'])],
                          [str(pdata['fn']), str(pdata['tp'])]],
                    texttemplate='%{text}', textfont={"size": 20}
                ))
                fig.update_layout(
                    title=f"{pname} - Karışıklık Matrisi",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# SAYFA: KLİNİK STRES TESTİ
# ══════════════════════════════════════════════════════════════════

elif page == " Klinik Stres Testi":
    st.markdown('<div class="hero-title" style="font-size:2em;"> Klinik Stres Testi Simülasyonu</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Şartname Sayfa 8: Test setinde Benign baskın dağılım</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Eğitim vs Test dağılımı karşılaştırması
    st.markdown("###  Eğitim vs Test Dağılımı")
    st.markdown("""
    <div class="glass-card">
        <p><strong>Şartnameye göre:</strong> Eğitim setinde Patojenik baskın (~%73), 
        test setinde ise Benign baskın (~%86) bir yapı kurgulanmıştır. 
        Bu, <strong>"Klinik Stres Testi"</strong> olarak adlandırılmaktadır.</p>
        <p style="color:#ff9800;"> Bu durum, modellerin gerçek dünya senaryosundaki 
        (düşük hastalık prevalansı) performansını ölçmek için tasarlanmıştır.</p>
    </div>
    """, unsafe_allow_html=True)

    # Dağılım grafikleri
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                       subplot_titles=("Eğitim Dağılımı (Master)", "Test Dağılımı (Master)"))

    fig.add_trace(go.Pie(
        labels=['Patojenik', 'Benign'],
        values=[2149, 782],
        marker_colors=['#ff1744', '#00e676'],
        hole=0.4
    ), row=1, col=1)

    fig.add_trace(go.Pie(
        labels=['Patojenik', 'Benign'],
        values=[500, 3000],
        marker_colors=['#ff1744', '#00e676'],
        hole=0.4
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Tüm panel test dağılımları
    st.markdown("### 📋 Panel Bazlı Test Dağılımları")

    panel_data = []
    for panel, dag in TEST_DAGILIM.items():
        total = dag['pathogenic'] + dag['benign']
        p_ratio = dag['pathogenic'] / total * 100
        panel_data.append({
            'Panel': panel,
            'Patojenik': dag['pathogenic'],
            'Benign': dag['benign'],
            'Toplam': total,
            'Patojenik %': f"{p_ratio:.1f}%",
            'Benign %': f"{100-p_ratio:.1f}%",
        })

    st.dataframe(pd.DataFrame(panel_data), use_container_width=True,
                 hide_index=True)

    # Stres testi raporu
    report_path = SONUC_DIR / 'test_raporu.txt'
    if report_path.exists():
        report = report_path.read_text(encoding='utf-8')
        import re
        stress_f1 = re.search(r'Stres Testi Ort F1:\s+([\d.]+)', report)
        stress_mcc = re.search(r'Stres Testi Ort MCC:\s+([\d.]+)', report)

        if stress_f1:
            st.markdown("### 🎯 Simülasyon Sonuçları")
            c1, c2 = st.columns(2)
            with c1:
                render_metric_card("Stres Testi F1", stress_f1.group(1),
                                   "Benign-baskın simülasyon")
            with c2:
                render_metric_card("Stres Testi MCC", stress_mcc.group(1) if stress_mcc else "—",
                                   "Benign-baskın simülasyon")


# ══════════════════════════════════════════════════════════════════
# SAYFA: HAKKINDA
# ══════════════════════════════════════════════════════════════════

elif page == " Hakkında":
    st.markdown('<div class="hero-title" style="font-size:2em;"> Hakkında</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00d4ff;">👥 Takım Bilgileri</h3>
            <table style="width:100%; color:#c0c0e0;">
                <tr><td><strong>Takım Adı:</strong></td><td>KGM_MED39</td></tr>
                <tr><td><strong>Üniversite:</strong></td><td>Kırklareli Üniversitesi</td></tr>
                <tr><td><strong>Bölüm:</strong></td><td>Yazılım Mühendisliği</td></tr>
            </table>
            <br>
            <div class="glass-card">
            <h3 style="color:#00d4ff;"> Ekibimiz</h3>
            <ul style="color:#c0c0e0;">
                <li><strong>Proje Lideri & Geliştirici:</strong> Arda Yiğit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#00d4ff;">🛠️ Kullanılan Teknolojiler</h3>
            <ul style="color:#c0c0e0;">
                <li><strong>XGBoost:</strong> Gradient Boosting</li>
                <li><strong>LightGBM:</strong> Microsoft Gradient Boosting</li>
                <li><strong>CatBoost:</strong> Yandex Gradient Boosting</li>
                <li><strong>ExtraTrees:</strong> Extremely Randomized Trees</li>
                <li><strong>Stacking:</strong> LogisticRegression Meta-Learner</li>
                <li><strong>Optuna:</strong> Bayesian Hiperparametre Optimizasyonu</li>
                <li><strong>SHAP:</strong> Açıklanabilir Yapay Zeka</li>
                <li><strong>Streamlit:</strong> Web Dashboard</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SHAP TAB
# ══════════════════════════════════════════════════════════════════
elif page == " YZ Açıklanabilirliği":
    st.markdown('<div class="hero-title" style="font-size:2em;"> Yapay Zeka Açıklanabilirliği (SHAP)</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Modelin kararlarını hangi genetik özelliklere göre aldığını gösteren şeffaflık analizi.</div>', unsafe_allow_html=True)

    shap_files = {
        'SHAP Özellik Önemi (Feature Importance)': 'shap_feature_importance.png',
        'SHAP Karar Dağılımı (Summary Plot)': 'shap_aciklanabilirlik_grafigi.png',
        'SHAP Bağımlılık Etkileşimleri (Dependence)': 'shap_dependence_plots.png',
    }

    for title, fname in shap_files.items():
        fpath = SONUC_DIR / fname
        if fpath.exists():
            with st.expander(f" {title}", expanded=False):
                st.image(str(fpath), use_container_width=True)

    st.markdown("""
    <div class="glass-card" style="text-align:center; margin-top:40px;">
        <p style="color:#7c4dff; font-size:1.2em; font-weight:600;">
             KGM_MED39 — Klinik Genomik Varyant Patojenite Tahmin Sistemi
        </p>
        <p style="color:#7c7ca0;">
            TEKNOFEST 2026 — Sağlıkta Yapay Zeka Yarışması — v4.0 Championship Edition
        </p>
    </div>
    """, unsafe_allow_html=True)
