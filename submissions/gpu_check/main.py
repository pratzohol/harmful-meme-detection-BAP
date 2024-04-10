# Standard Python Library
import sys
import multiprocessing
import time

import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import torch


if __name__ == "__main__":
    file_paths = []

    if torch.cuda.is_available():
        sys.exit()
    else:
        # Iteration loop to get new image filepath from sys.stdin:
        for line in sys.stdin:
            # IMPORTANT: Please ensure any trailing whitespace (eg: \n) is removed. This may impact some modules to open the filepath
            image_path = line.rstrip()
            file_paths.append(image_path)


        try:
            for fp in file_paths:
                time.sleep(1)
                proba, label = 1, 0
                sys.stdout.write(f"{proba:.4f}\t{label}\n")

        except Exception as e:
            # Output to any raised/caught error/exceptions to stderr
            sys.stderr.write(str(e))
