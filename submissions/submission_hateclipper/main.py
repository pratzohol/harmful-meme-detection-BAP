# Standard Python Library
import sys
import multiprocessing

import cv2
import pytesseract
from pytesseract import Output
import numpy as np


def preprocess_image(im):
    """Summary

    Args:
        im (np.array): Image in BGR format after using cv2.imread(<filePath>)

    Returns:
        np.array :
    """
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.bilateralFilter(im, 9, 55, 60)
    _, im = cv2.threshold(im, 235, 255, cv2.THRESH_BINARY_INV)
    return im


def extract_text_from_meme(im):
    im = preprocess_image(im)

    tess_config = r'-l eng+chi_sim+chi_tra+tam+msa --tessdata-dir /usr/share/tesseract-ocr/tessdata_best --oem 1 --psm 11'
    txt = pytesseract.image_to_string(im, config=tess_config)
    txt = txt.replace('\n', ' ').strip()
    return txt


def process_line_by_line(filepath):
    im = cv2.imread(filepath)
    text = extract_text_from_meme(im)
    return text


if __name__ == "__main__":
    file_paths = []

    # Iteration loop to get new image filepath from sys.stdin:
    for line in sys.stdin:
        # IMPORTANT: Please ensure any trailing whitespace (eg: \n) is removed. This may impact some modules to open the filepath
        image_path = line.rstrip()
        file_paths.append(image_path)

    try:
        with multiprocessing.Pool(4) as pool:
            texts = pool.map(process_line_by_line, file_paths)

        for txt in texts:
            proba, label = 1, 0
            sys.stdout.write(f"{proba:.4f}\t{label}\n")

    except Exception as e:
        # Output to any raised/caught error/exceptions to stderr
        sys.stderr.write(str(e))
