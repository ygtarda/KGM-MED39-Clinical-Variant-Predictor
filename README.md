# 🧬 KGM_MED39 - Klinik Genomik Varyant Patojenite Tahmin Sistemi (v4.0)

> **TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Üniversite ve Üzeri Seviyesi**  
> **Takım:** KGM_MED39 | **Üniversite:** Kırklareli Üniversitesi, Yazılım Mühendisliği  
> **Üyeler:** Enes Dolgun, Enes Turan, Arda Yiğit, Bekir Berk Kahveci, Mehmet Akif Gök

---

## 📋 İçindekiler

1. [Proje Özeti](#proje-özeti)
2. [Problem Tanımı](#problem-tanımı)
3. [Veri Seti Açıklaması](#veri-seti-açıklaması)
4. [Sistem Mimarisi](#sistem-mimarisi)
5. [Feature Engineering (Özellik Mühendisliği)](#feature-engineering)
6. [Model Mimarisi ve Yöntem](#model-mimarisi-ve-yöntem)
7. [Hiperparametre Optimizasyonu](#hiperparametre-optimizasyonu)
8. [Klinik Stres Testi Simülasyonu](#klinik-stres-testi-simülasyonu)
9. [Çapraz Doğrulama ve Stacking](#çapraz-doğrulama-ve-stacking)
10. [Karar Eşiği (Threshold) Optimizasyonu](#karar-eşiği-optimizasyonu)
11. [Panel Testleri](#panel-testleri)
12. [SHAP Açıklanabilirlik](#shap-açıklanabilirlik)
13. [Web Arayüzü (Streamlit)](#web-arayüzü)
14. [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
15. [Sonuçlar ve Bulgular](#sonuçlar-ve-bulgular)
16. [Klasör Yapısı](#klasör-yapısı)
17. [PDR Raporu İçin Notlar](#pdr-raporu-için-notlar)
18. [Kaynakça](#kaynakça)

---

## Proje Özeti

Bu proje, **missense varyantların** patojenite sınıflandırması (Patojenik/Benign) için geliştirilmiş ileri düzey bir makine öğrenmesi pipeline'ıdır. ACMG (American College of Medical Genetics and Genomics) rehberlerine uygun etiketleme kullanılmaktadır.

### Temel Yenilikler (v4.0):
1. **Stacking Ensemble** — 4 base model + LogisticRegression meta-learner
2. **Klinik Stres Testi Simülasyonu** — Benign-baskın test dağılımı CV sırasında simüle edilir
3. **Tüm modeller için Optuna** — XGBoost, LightGBM, CatBoost, ExtraTrees tam optimizasyon
4. **F1 + MCC Kompozit Threshold** — Tek metrik yerine iki metriği dengeleyen karar eşiği
5. **Panel-Specific Threshold** — Her panel için ayrı optimize edilmiş karar eşikleri
6. **Streamlit Web Dashboard** — Profesyonel, interaktif demo arayüzü
7. **Modüler Kod Yapısı** — `src/` paketi ile DRY prensibi

---

## Problem Tanımı

**Missense varyant patojenite tahmini**, klinik genetikte en önemli sorunlardan biridir. İnsan genomundaki tek nükleotid varyantları (SNV) amino asit değişimlerine yol açabilir. Bu değişimlerin bir kısmı hastalığa neden olurken (patojenik), bir kısmı zararsızdır (benign). Klinik laboratuvarlarda, "Variant of Uncertain Significance" (VUS — önemi belirsiz varyant) olarak sınıflandırılan binlerce varyant, hastaların genetik tanılarını geciktirir.

**Tablo türü veri yapılarında sınıf dengesizliği** bu problemin temel zorluğudur:
- Eğitim setinde Patojenik varyantlar ağırlıktadır (~%73)
- Gerçek klinik pratikte (ve yarışma test setinde) Benign varyantlar baskındır (~%86)
- Bu durum, eğitim ile test arasında ciddi bir dağılım kayması (distribution shift) yaratır
- Modelin Benign vakaları doğru tanıyamaması, yanlış pozitif (False Positive) artışına yol açar

### Klinik Önemi:
- **Yanlış Pozitif (FP):** Sağlıklı bir varyantı patojenik olarak etiketlemek → gereksiz klinik takip, hasta anksiyetesi
- **Yanlış Negatif (FN):** Patojenik bir varyantı benign olarak etiketlemek → kaçırılan tanı, tedavi gecikmesi
- Her iki hata türü de klinik açıdan ciddi sonuçlar doğurur

---

## Veri Seti Açıklaması

### Veri Kaynakları
- **Patojenik sınıf:** ClinVar ve ClinGen veri tabanlarından, "Expert Panel" ve güvenilir "Practice Guideline" inceleme statüsüne sahip, 3-4 yıldız güvenilirlik seviyesindeki missense varyantlar (3062 varyant)
- **Benign sınıf:** ClinVar (1381 varyant) + gnomAD sağlıklı popülasyon varyantları (2153 varyant)

### Veri Setleri

| Panel | Eğitim (P/B) | Test Tahmini (P/B) | Patojenik Oranı (Test) |
|-------|--------------|---------------------|----------------------|
| MASTER | 2149/782 | 500/3000 | %14.3 |
| KANSER | 268/120 | 100/500 | %16.7 |
| PAH | 310/62 | 100/250 | %28.6 |
| CFTR | 90/21 | 20/100 | %16.7 |

### Öznitelik Kategorileri (353 kolon)
- **AL_ (Frekans/Popülasyon):** Allel frekansları, popülasyon yaygınlığı
- **EK_ (Evrimsel Korunmuşluk):** Filogenetik korunmuşluk skorları
- **CAT_ (Kategorik Meta-Veri):** Popülasyon etiketleri, kalite bayrakları, arkaik genom bilgisi
- **AA_ (Amino Asit Değişimi):** Referans ve alternatif amino asit bilgileri

> **Not:** Tersine mühendisliği engellemek amacıyla orijinal kolon isimleri gizlenmiştir. Genomik adres bilgileri kaldırılmıştır.

---

## Sistem Mimarisi

```
                    ┌─────────────────────────┐
                    │     HAM VERİ (CSV)       │
                    │    4 Panel × 353 Kolon   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  FEATURE ENGINEERING     │
                    │  (src/feature_eng...)    │
                    │  • AL İstatistikleri     │
                    │  • EK Korunmuşluk       │
                    │  • AA Biyokimya         │
                    │  • Etkileşim Terimleri   │
                    │  • Eksik Veri Pattern    │
                    │  → 499+ özellik          │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  OPTUNA OPTİMİZASYON    │
                    │  Bayesian TPE Sampler    │
                    │  60 trial × 4 model     │
                    └───────────┬─────────────┘
                                │
           ┌──────────┬─────────▼──────┬──────────┐
           │          │                │          │
      ┌────▼───┐ ┌───▼────┐ ┌────▼─────┐ ┌───▼──────┐
      │XGBoost │ │LightGBM│ │ CatBoost │ │ExtraTrees│
      │(Level 1)│ │(Level 1)│ │(Level 1)│ │(Level 1)│
      └────┬───┘ └───┬────┘ └────┬─────┘ └───┬──────┘
           │         │           │           │
           └─────────┴──────┬────┴───────────┘
                            │
                 ┌──────────▼──────────┐
                 │  OOF PREDICTIONS    │
                 │  (Out-of-Fold)      │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  STACKING           │
                 │  META-LEARNER       │
                 │  (LogisticRegression)│
                 │  (Level 2)          │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  F1+MCC KOMPOZİT    │
                 │  THRESHOLD OPT.     │
                 │  (Panel-Specific)    │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  TAHMİN ÇIKTISI     │
                 │  Patojenik / Benign │
                 └─────────────────────┘
```

---

## Feature Engineering

Toplam **499+ mühendislik özelliği** üretilmektedir. Detaylar:

### 1. AL (Frekans/Popülasyon) İstatistikleri
| Özellik | Açıklama | Domain Bilgisi |
|---------|----------|---------------|
| FE_AL_mean | Ortalama frekans | Yüksek = Benign |
| FE_AL_std | Frekans standart sapma | Düşük varyans = Güvenilir |
| FE_AL_min, max, range | Min/max/aralık | Geniş aralık = Şüpheli |
| FE_AL_skew, kurtosis | Dağılım şekli | Çarpıklık analizi |
| FE_AL_entropy | Frekans entropisi | Bilgi yoğunluğu |
| FE_AL_high_freq_count | >0.01 frekans sayısı | Sık = Benign olasılığı yüksek |
| FE_AL_rare_count | 0 < f < 0.001 | Nadir = Patojenik olasılığı yüksek |
| FE_AL_cv | Varyasyon katsayısı | Tutarlılık ölçümü |

### 2. EK (Evrimsel Korunmuşluk) İstatistikleri
| Özellik | Açıklama | Domain Bilgisi |
|---------|----------|---------------|
| FE_EK_mean | Ortalama korunmuşluk | Yüksek = Patojenik |
| FE_EK_high_cons | >0.8 skor sayısı | Çok korunmuş bölge |
| FE_EK_low_cons | <0.2 skor sayısı | Az korunmuş bölge |
| FE_EK_pos_ratio | Pozitif skor oranı | Genel korunmuşluk eğilimi |

### 3. Amino Asit Biyokimyasal Özellikleri
| Özellik | Açıklama | Domain Bilgisi |
|---------|----------|---------------|
| FE_AA_same | Aynı AA değişimi | Sinonim varyantlar |
| FE_AA_hydro_change | Hidrofobiklik değişimi | Protein katlanma etkisi |
| FE_AA_charge_change | Yük değişimi | Elektrostatik etkileşim |
| FE_AA_size_change | Boyut değişimi | Sterik engel |
| FE_AA_polarity_change | Polarite değişimi | Çözünürlük etkisi |
| FE_AA_radical | Toplam radikal skor | Grantham benzeri şiddet |
| FE_radical_x_conservation | Radikal × Korunmuşluk | Çapraz domain sinyal |

### 4. Etkileşim Terimleri (SHAP yönlendirmeli)
- EK çiftleri arası çarpım, fark ve oran (12 çift × 3 işlem = 36 özellik)
- EK-AL çapraz etkileşimler (düşük frekans + yüksek korunmuşluk)

### 5. Eksik Veri Pattern
Eksik veri miktarı ve oranı kendi başına bilgilendiricidir — bazı varyant tiplerinde belirli alanlar sistematik olarak eksik olabilir.

---

## Model Mimarisi ve Yöntem

### Base Modeller (Level 1)

| Model | Özellik | Neden Seçildi? |
|-------|---------|---------------|
| **XGBoost** | Gradient Boosting | Tablo verilerinde SOTA, hız/performans dengesi |
| **LightGBM** | Leaf-wise büyüme | Yüksek boyutlu veri, düşük bellek |
| **CatBoost** | Ordered boosting | Kategorik veri desteği, overfitting direnci |
| **ExtraTrees** | Extremely Randomized | Randomizasyon ile varyans azaltma |

### Meta-Learner (Level 2) — Stacking
- **LogisticRegression** (L2 regularization, class_weight='balanced')
- Base modellerin OOF (Out-of-Fold) tahminlerini girdi olarak alır
- Model çeşitliliğinden faydalanarak genelleme gücünü artırır
- Farklı modellerin farklı varyant tiplerindeki güçlü yönlerini birleştirir

### Neden Stacking?
Basit ağırlıklı oylama (voting) sabit ağırlıklar kullanır. Stacking ise:
- **Dinamik ağırlıklandırma:** Her varyant için en uygun model kombinasyonunu öğrenir
- **Hata korelasyonu:** Modellerin hata yaptığı durumları diğer modellerin telafi etmesini sağlar
- **Bias-variance dengesi:** Overfitting'e karşı daha dirençli

---

## Hiperparametre Optimizasyonu

**Optuna** kütüphanesi ile **Bayesian TPE (Tree-structured Parzen Estimator)** kullanılır.

### Optimizasyon Detayları
- **Trial sayısı:** 60 trial/model (XGB, LGBM) + 40 trial (CatBoost) + 30 trial (ExtraTrees)
- **Timeout:** 600 saniye/model
- **Sampler:** TPESampler (seed=42, tekrarlanabilirlik)
- **CV:** 3-Fold StratifiedKFold (optimizasyon hızı için)
- **Hedef metrik:** F1 + MCC kompozit skor (0.6×F1 + 0.4×((MCC+1)/2))

### Arama Uzayları

**XGBoost:**
```
n_estimators:    [200, 800]
learning_rate:   [0.01, 0.1] (log)
max_depth:       [4, 10]
min_child_weight:[1, 10]
subsample:       [0.6, 1.0]
colsample_bytree:[0.5, 1.0]
reg_alpha:       [1e-8, 10.0] (log)
reg_lambda:      [1e-8, 10.0] (log)
gamma:           [1e-8, 5.0] (log)
```

**LightGBM:**
```
n_estimators:    [200, 800]
learning_rate:   [0.01, 0.1] (log)
max_depth:       [4, 10]
num_leaves:      [20, 127]
min_child_samples:[5, 50]
subsample:       [0.6, 1.0]
colsample_bytree:[0.5, 1.0]
reg_alpha:       [1e-8, 10.0] (log)
reg_lambda:      [1e-8, 10.0] (log)
```

**CatBoost:**
```
iterations:      [200, 800]
learning_rate:   [0.01, 0.1] (log)
depth:           [4, 10]
l2_leaf_reg:     [1e-3, 10.0] (log)
bagging_temperature: [0.0, 1.0]
random_strength: [1e-3, 10.0] (log)
```

**ExtraTrees:**
```
n_estimators:    [200, 800]
max_depth:       [6, 20]
min_samples_split:[2, 20]
min_samples_leaf: [1, 10]
max_features:    [sqrt, log2, None]
```

---

## Klinik Stres Testi Simülasyonu

### Problem
Eğitim: %73 Patojenik, %27 Benign  
Test: %14 Patojenik, %86 Benign

Bu "dağılım kayması" (covariate shift), eğitimde yüksek performans gösteren modelin testte başarısız olmasına neden olabilir.

### Çözüm
Çapraz doğrulama sırasında her validation fold'unda:
1. Validation setindeki sınıf dağılımı tespit edilir
2. Benign örnekler oversampling ile artırılarak ~%86 oranına getirilir
3. Bu resampled validation set üzerinde threshold optimize edilir
4. Model, test dağılımına benzer koşullarda değerlendirilir

Bu sayede:
- False Positive oranı minimize edilir
- Model, Benign-baskın ortamda yüksek performans gösterecek şekilde kalibre edilir

---

## Çapraz Doğrulama ve Stacking

### CV Stratejisi
- **RepeatedStratifiedKFold:** 5-Fold × 2-Repeat = 10 fold
- Stratified: Sınıf dengesini her fold'da korur
- Repeated: Varyans tahminini iyileştirir

### Stacking Süreci
1. Her fold'da 4 base model eğitilir
2. Validation tahminleri OOF matrisine yazılır
3. Tüm fold'lar bitince OOF matrisi oluşur (N×4 boyutlu)
4. LogisticRegression bu matris üzerinde eğitilir (Meta-Learner)
5. Final modeli tüm veri üzerinde eğitilir

### Overfitting Önlemleri
- **L2 Regularization:** Tüm modellerde ve meta-learner'da
- **class_weight='balanced':** ExtraTrees ve meta-learner'da
- **scale_pos_weight:** XGBoost ve LightGBM'de sınıf dengesizliği telafisi
- **auto_class_weights='Balanced':** CatBoost'ta otomatik
- **Repeated CV:** Tek fold'a bağımlılığı azaltır
- **Early stopping (Optuna):** Overfitting trial'ları erken keser

---

## Karar Eşiği Optimizasyonu

### Neden 0.5 Değil?
Standart 0.5 threshold, dengeli veri setleri için uygundur. Ancak:
- Eğitim setimiz Patojenik-baskın (%73)
- Test setimiz Benign-baskın (%86)
- Bu durumda 0.5 threshold çok fazla False Positive üretir

### Kompozit Metrik
```
Score = 0.6 × F1 + 0.4 × ((MCC + 1) / 2)
```
- **F1:** Ana sıralama metriği (şartname)
- **MCC:** Dengesiz veri setlerinde daha güvenilir (şartname ek metrik)
- MCC [-1, 1] aralığından [0, 1]'e normalize edilir

### Panel-Specific Threshold
Her panel (KANSER, PAH, CFTR) farklı genetik karakteristiğe sahip:
- CFTR gen paneli: Kistik fibrozis odaklı, daha az varyant
- PAH gen paneli: Fenilketonüri odaklı, yüksek patojenik oranı
- KANSER: Herediter kanser genleri, çeşitli genler

Her panel için ayrı threshold optimize edilerek panel-specific performans maksimize edilir.

---

## Panel Testleri

### Veri Sızıntısı (Leakage) Koruması
Master seti üzerinde eğitilen model panel setleri üzerinde test edilirken:
1. **Tüm veri testi:** Referans sonuç (overlap dahil)
2. **Bağımsız test:** Overlap çıkarılmış (Master'da olmayan varyantlar)
3. **Klinik stres testi bilgisi:** Şartname test dağılımı raporlanır

Bağımsız test sonuçları, modelin gerçek genelleme gücünü gösterir.

---

## SHAP Açıklanabilirlik

### Neden SHAP?
- **Şartname gerekliliği:** Modellerin karar mekanizması açıklanabilir olmalı
- **Klinik güvenilirlik:** Doktorların modele güvenmesi için kararların yorumlanabilir olması gerekir
- **Feature engineering doğrulama:** Hangi özelliklerin gerçekten önemli olduğunu gösterir

### Üretilen Grafikler
1. **Summary Plot:** Her özelliğin model kararına etkisi (yön + büyüklük)
2. **Feature Importance Bar:** Mean |SHAP| değerine göre sıralama
3. **Dependence Plots:** Top 6 özelliğin nonlinear ilişkileri

### Yorumlama
- **Sağ (pozitif SHAP):** Patojenik yönünde etki
- **Sol (negatif SHAP):** Benign yönünde etki
- **Kırmızı noktalar:** Yüksek değerli örnekler
- **Mavi noktalar:** Düşük değerli örnekler

---

## Web Arayüzü

### Streamlit Professional Dashboard
Finalde jüri önünde kullanılacak interaktif web arayüzü:

| Sayfa | İçerik |
|-------|--------|
| 🏠 Ana Sayfa | Proje özeti, mimari diyagram, güncel metrikler |
| 🔬 Varyant Tahmin | CSV yükleme, gerçek zamanlı tahmin, sonuç indirme |
| 📊 Model Performans | F1/MCC/PR-AUC grafikleri, fold karşılaştırma |
| 🧪 Panel Analizi | Panel bazlı performans, karışıklık matrisleri |
| 📈 Klinik Stres Testi | Dağılım karşılaştırma, simülasyon sonuçları |
| ℹ️ Hakkında | Takım bilgileri, teknolojiler, SHAP grafikleri |

### Tasarım Özellikleri
- Dark mode glassmorphism tasarım
- İnteraktif Plotly grafikleri
- Responsive layout
- CSV drag-and-drop yükleme
- Sonuç CSV indirme

---

## Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```
> **macOS için:** `brew install libomp` gereklidir.

### 2. Model Eğitimi (Pipeline Oluşturma)
```bash
python kgm_med39_predictor.py
```
> Bu işlem Optuna optimizasyonu nedeniyle ~15-20 dakika sürebilir.

### 3. Web Dashboard'u Başlat
```bash
streamlit run app.py
```
> Tarayıcıda `http://localhost:8501` adresinde açılır.

### 4. Yarışma Günü Tahmin Üretimi
```bash
# Genel (Master threshold ile)
python tahmin_uret.py --input test_verisi.csv --output sonuclar.csv

# Panel-specific threshold ile
python tahmin_uret.py --input test_kanser.csv --output sonuc_kanser.csv --panel KANSER
```

### 5. Shell Script ile Çalıştırma
```bash
# Model eğitimi
./calistir.sh

# Tahmin üretimi
./tahmin_et.sh test_verisi.csv sonuclar.csv
```

---

## Sonuçlar ve Bulgular

### Master Panel (OOF)
| Metrik | Değer |
|--------|-------|
| F1 Score | *Eğitim sonrası güncellenir* |
| MCC | *Eğitim sonrası güncellenir* |
| PR-AUC | *Eğitim sonrası güncellenir* |
| Precision | *Eğitim sonrası güncellenir* |
| Recall | *Eğitim sonrası güncellenir* |

> **Not:** Detaylı sonuçlar `sonuclar/test_raporu.txt` dosyasında bulunmaktadır.

### Panel Sonuçları
Tüm panel sonuçları (KANSER, PAH, CFTR) `test_raporu.txt` içinde yer almaktadır.

### Klinik Stres Testi Sonuçları
Benign-baskın simülasyon sonuçları da raporda yer almaktadır.

---

## Klasör Yapısı

```text
KGM_MED39_Variant_Predictor/
│
├── src/                               # Modüler kaynak kodu
│   ├── __init__.py
│   ├── config.py                      # Merkezi konfigürasyon
│   ├── feature_engineering.py         # Ortak FE modülü
│   └── utils.py                       # Yardımcı fonksiyonlar
│
├── veriler/                           # Yarışma CSV verileri
│   ├── YARISMA_TRAIN_MASTER.csv
│   ├── YARISMA_TRAIN_KANSER.csv
│   ├── YARISMA_TRAIN_PAH.csv
│   └── YARISMA_TRAIN_CFTR.csv
│
├── dokumanlar/                        # Yarışma belgeleri
│   ├── sartname.pdf
│   └── pdr_sablon.docx
│
├── sonuclar/                          # Otomatik oluşan çıktılar
│   ├── kgm_med39_pipeline.joblib      # Ana tahmin pipeline'ı
│   ├── test_raporu.txt                # Detaylı metrik raporu
│   ├── shap_*.png                     # Açıklanabilirlik grafikleri
│   ├── fold_karsilastirma.png         # CV fold grafikleri
│   ├── precision_recall_egrisi.png    # PR eğrisi
│   ├── karisiklik_matrisi.png         # Confusion matrix
│   ├── veri_imputer.joblib            # Imputer
│   ├── veri_scaler.joblib             # Scaler
│   ├── model_kolonlari.joblib         # Kolon isimleri
│   └── freq_maps.joblib              # Frekans haritaları
│
├── kgm_med39_predictor.py             # Model Eğitim / Raporlama
├── tahmin_uret.py                     # Final Günü Tahmin Scripti
├── app.py                             # Streamlit Web Dashboard
├── requirements.txt                   # Python bağımlılıkları
├── calistir.sh                        # Model eğitim script
├── tahmin_et.sh                       # Tahmin üretim script
└── README.md                          # Bu dosya
```

---

## PDR Raporu İçin Notlar

> **Bu bölüm, PDR raporunu yazacak kişi için hazırlanmıştır.**

### 1. GİRİŞ (10 puan)
**Yazılması gerekenler:**
- Missense varyant patojenite problemi tanımı (yukarıdaki "Problem Tanımı" bölümü)
- Klinik ve genomik bağlamdaki önemi (VUS problemi, ACMG rehberleri)
- Sınıf dengesizliği probleminin model başarımına etkisi
- Güncel 5-10 uluslararası çalışma:
  1. Rentzsch et al. (2019) - CADD: Combined Annotation Dependent Depletion
  2. Ioannidis et al. (2016) - REVEL: Rare Exome Variant Ensemble Learner
  3. Vaser et al. (2016) - SIFT: Sorting Intolerant From Tolerant
  4. Adzhubei et al. (2010) - PolyPhen-2
  5. Schwarz et al. (2014) - MutationTaster2
  6. Li et al. (2020) - ClinPred
  7. Qi et al. (2021) - MVPmeta
  8. Sundaram et al. (2018) - PrimateAI
  9. Cheng et al. (2023) - AlphaMissense (Google DeepMind)
  10. Brandes et al. (2023) - EVE (Evolutionary Model of Variant Effect)

### 2. YÖNTEM (25 puan)
**Yazılması gerekenler:**
- Veri kümesi yapısı (353 kolon, 4 panel, şifreli kolon isimleri)
- Eksik değer yönetimi: Median imputation (SimpleImputer), >%70 eksik kolon silme
- Aykırı değer: RobustScaler (IQR bazlı, aykırı değerlere dirençli)
- Dış kaynak: Eklenmedi (şartname gereği)
- Feature engineering detayları (yukarıdaki "Feature Engineering" bölümü — 499+ özellik)
- Model seçim gerekçeleri (yukarıdaki "Model Mimarisi" bölümü)
- Hiperparametre optimizasyonu (Optuna Bayesian TPE, detaylar yukarıda)
- Çapraz doğrulama: RepeatedStratifiedKFold (5×2=10 fold)
- Overfitting önlemleri (L2 reg, class_weight, repeated CV)
- Açıklanabilirlik: SHAP TreeExplainer
- Karar eşiği: F1+MCC kompozit threshold optimizasyonu

### 3. BULGULAR (30 puan)
**Yazılması gerekenler:**
- `sonuclar/test_raporu.txt` dosyasındaki tüm metrikler
- Panel bazlı ayrı ayrı F1, MCC, PR-AUC değerleri
- Karışıklık matrisleri (`sonuclar/karisiklik_matrisi.png`)
- Fold karşılaştırma grafikleri (`sonuclar/fold_karsilastirma.png`)
- PR eğrisi (`sonuclar/precision_recall_egrisi.png`)
- SHAP grafikleri (`sonuclar/shap_*.png`)
- Farklı karar eşiklerinin karşılaştırması (threshold analizi)
- Meta-Learner vs Ağırlıklı Voting karşılaştırması

### 4. SONUÇ (25 puan)
**Yazılması gerekenler:**
- Modelin güçlü yönleri: Stacking ensemble, domain bilgisi ile FE, stres testi dayanıklılığı
- Modelin zayıf yönleri: Genomik adres bilgisi eksikliği, küçük veri seti
- FP analizi: Hangi varyant tiplerinde yanlış pozitif üretiliyor? (SHAP'tan yararlanılabilir)
- FN analizi: Hangi patojenik varyantlar kaçırılıyor?
- Klinik anlam: FP = gereksiz klinik takip, FN = kaçırılan tanı
- Literatürdeki yeri: CADD, REVEL ile karşılaştırma
- Yarışma son basamağında zorluklar: Benign-baskın test seti, domain shift

### 5. KAYNAKÇA (10 puan — Rapor Düzeni dahil)
- IEEE formatında kaynakça
- Aptos yazı tipi, 12 punto, 1.15 satır aralığı
- Maksimum 10 sayfa (kapak ve içindekiler hariç)
- İki tarafa yaslı, kenar boşlukları: üst 2.8, alt-sağ-sol 2.5

---

## Kaynakça

[1] P. Rentzsch et al., "CADD-Splice — improving genome-wide variant effect prediction using deep learning-derived splice scores," *Genome Medicine*, vol. 13, no. 1, 2021.

[2] N. M. Ioannidis et al., "REVEL: An ensemble method for predicting the pathogenicity of rare missense variants," *American Journal of Human Genetics*, vol. 99, no. 4, pp. 877–885, 2016.

[3] R. Vaser et al., "SIFT missense predictions for genomes," *Nature Protocols*, vol. 11, no. 1, pp. 1–9, 2016.

[4] I. A. Adzhubei et al., "A method and server for predicting damaging missense mutations," *Nature Methods*, vol. 7, no. 4, pp. 248–249, 2010.

[5] J. M. Schwarz et al., "MutationTaster2: mutation prediction for the deep-sequencing age," *Nature Methods*, vol. 11, no. 4, pp. 361–362, 2014.

[6] J. Cheng et al., "Accurate proteome-wide missense variant effect prediction with AlphaMissense," *Science*, vol. 381, no. 6664, 2023.

[7] N. Brandes et al., "Genome-wide prediction of disease variant effects with a deep protein language model," *Nature Genetics*, vol. 55, pp. 1512–1522, 2023.

[8] S. Richards et al., "Standards and guidelines for the interpretation of sequence variants," *Genetics in Medicine*, vol. 17, no. 5, pp. 405–423, 2015. (ACMG Standards)

[9] T. Akiba et al., "Optuna: A next-generation hyperparameter optimization framework," *KDD*, 2019.

[10] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, 2017. (SHAP)

---

**Takım:** KGM_MED39 | **Versiyon:** v4.0 Championship Edition  
**Son Güncelleme:** 2026-06-26
