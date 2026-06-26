#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Kullanim Hatalari!"
    echo "Dogru Kullanim: ./tahmin_et.sh <girdi_test_verisi.csv> <cikti_tahmin.csv>"
    echo "Ornek: ./tahmin_et.sh veriler/test_seti.csv benim_sonucum.csv"
    exit 1
fi

INPUT_FILE=$1
OUTPUT_FILE=$2

echo "=========================================================="
echo " KGM_MED39 - Otonom Tahmin Araci"
echo " Test Verisi: $INPUT_FILE"
echo "=========================================================="

python3 tahmin_uret.py --input "$INPUT_FILE" --output "$OUTPUT_FILE"
