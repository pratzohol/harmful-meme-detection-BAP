#!/bin/bash

# Define the directory where Tesseract puts its language data files
TESSDATA_DIR="/usr/share/tesseract-ocr/tessdata/"

# Download English traineddata
wget -P "$TESSDATA_DIR" https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

# Download Simplified Chinese traineddata
wget -P "$TESSDATA_DIR" https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata

# Download Traditional Chinese traineddata
wget -P "$TESSDATA_DIR" https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_tra.traineddata

# Download Tamil traineddata
wget -P "$TESSDATA_DIR" https://github.com/tesseract-ocr/tessdata_best/raw/main/tam.traineddata

# Download Malay trainneddata
wget -P "$TESSDATA_DIR" https://github.com/tesseract-ocr/tessdata_best/raw/main/msa.traineddata