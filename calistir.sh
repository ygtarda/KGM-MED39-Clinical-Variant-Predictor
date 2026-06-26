#!/bin/bash
echo "=========================================================="
echo " KGM_MED39 - Model Egitim Pipeline'i Basliyor"
echo " Lutfen bekleyin, Optuna optimizasyonu yaklasik 10 dk surebilir."
echo "=========================================================="
python3 kgm_med39_predictor.py
echo "Islem Tamamlandi! Sonuclari 'sonuclar/' klasorunde bulabilirsiniz."
