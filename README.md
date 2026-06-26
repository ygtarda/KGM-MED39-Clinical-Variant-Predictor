# 🧬 KGM_MED39 - Klinik Genomik Varyant Patojenite Tahmin Sistemi (v4.0)

> **TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması** Üniversite ve Üzeri Kategorisi için geliştirilmiş profesyonel yapay zeka karar destek sistemidir.

Bu proje, genomik varyantların patojenite (hastalık yapıcı) durumlarını yüksek hassasiyetle tahmin etmek amacıyla geliştirilmiştir. Stacking Meta-Learner mimarisi ve "Klinik Stres Testi" altyapısı kullanılarak %90+ başarı oranlarına ulaşılmıştır.

---

## 🚀 Proje Klasör Yapısı

Jürinin ve değerlendiricilerin projeyi çalıştırması için tasarlanan klasör yapısı:

```text
KGM-MED39-Clinical-Variant-Predictor/
├── src/                      # Konfigürasyon, Veri Mühendisliği ve Yardımcı Kodlar
├── sonuclar/                 # Eğitilmiş Model Ağırlıkları (.joblib) ve SHAP Grafikleri
├── app.py                    # Streamlit Web Arayüzü / Dashboard
├── kgm_med39_predictor.py    # Modeli Sıfırdan Eğiten Ana Yapay Zeka Kodu
├── tahmin_uret.py            # Yarışma Günü Toplu Tahmin ve Test Kodu
├── requirements.txt          # Gerekli Kütüphaneler
└── egitim.log                # 45 Dakikalık Detaylı Optuna Eğitim Logları
```

---

## 🛠️ Nasıl Çalıştırılır?

Projeyi çalıştırmak için aşağıdaki adımları sırasıyla uygulayınız.

### 1. Gerekli Kütüphanelerin Kurulumu
Projenin kök dizininde bir terminal açın ve aşağıdaki komutu çalıştırarak gerekli tüm Python kütüphanelerini yükleyin:
```bash
pip install -r requirements.txt
```

### 2. Arayüzün (Dashboard) Başlatılması
Yapay zeka modelinin tüm özelliklerini interaktif olarak incelemek, SHAP grafiklerini görmek ve tekli/çoklu tahmin yapmak için Streamlit arayüzünü başlatın:
```bash
streamlit run app.py
```
*(Bu komut varsayılan web tarayıcınızda `http://localhost:8501` adresini otomatik olarak açacaktır.)*

### 3. Yarışma Günü Toplu Test (Terminal Üzerinden)
Yarışma sırasında verilecek gizli test setini (CSV formatında) hızlıca test etmek için `tahmin_uret.py` dosyasını kullanabilirsiniz:

**Genel Tahmin İçin:**
```bash
python tahmin_uret.py --input test_verisi.csv --output sonuclar.csv
```

**Belirli Bir Panele Özel Tahmin İçin (Örn: KANSER, PAH, CFTR):**
```bash
python tahmin_uret.py --input test_verisi.csv --output sonuclar.csv --panel KANSER
```

---

## 🧠 Model Mimarisi ve v4.0 Skorları

Sistem, 4 farklı gelişmiş algoritmanın (XGBoost, LightGBM, CatBoost, ExtraTrees) birleşiminden oluşan **Stacking Meta-Learner (Logistic Regression)** mimarisini kullanmaktadır. Optuna ile hiperparametreleri 45 dakika boyunca optimize edilmiştir.

### 🏆 Kesinleşmiş OOF (Out-of-Fold) Performans Metrikleri:
- **Ortalama F1 Skoru:** 0.8941
- **Ortalama PR-AUC:** 0.9276
- **Ortalama MCC:** 0.5552
- **Recall (Hassasiyet):** 0.9448 (Patojenik varyantları kaçırmama oranı)
- **Precision (Kesinlik):** 0.8497

*(Detaylı Optuna hiperparametre denemeleri ve 10-Fold CV sonuçları projedeki `egitim.log` dosyasında mevcuttur.)*

---

## 👨‍💻 Geliştirici
**Proje Lideri & Geliştirici:** Arda Yiğit
