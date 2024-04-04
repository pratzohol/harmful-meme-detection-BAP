# Standard Python Library
import sys
import logging
import random
from tqdm import trange
from tqdm import tqdm

import cv2
import pytesseract
from pytesseract import Output
import numpy as np
# import torch

import multiprocessing

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
    txt = txt.replace('\n\n', '\n').rstrip()

    d = pytesseract.image_to_data(im, output_type=Output.DICT, config=tess_config)
    n_boxes = len(d["level"])
    coordinates = []

    for i in range(n_boxes):
        (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
        coordinates.append((x, y, w, h))
    return txt, coordinates[1:]


def get_image_mask(image, coordinates_to_mask):
    # Create a mask image with image_size
    image_mask = np.zeros_like(image[:, :, 0])

    for coordinates in coordinates_to_mask:
        # unpack the coordinates
        x, y, w, h = coordinates

        # set mask to 255 for coordinates
        image_mask[y : y + h, x : x + w] = 255

    return image_mask


def get_image_inpainted(image, image_mask):
    # Perform image inpainting to remove text from the original image
    image_inpainted = cv2.inpaint(image, image_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return image_inpainted


def process_line_by_line(filepath):
    # 1. Open image filepath ========================================= #
    im = cv2.imread(filepath)

    # 2. Get meme text =============================================== #
    text, coordinates = extract_text_from_meme(im)

    # 3. Get inpainting ============================================== #
    # Get image mask for image inpainting
    im_mask = get_image_mask(image=im, coordinates_to_mask=coordinates)

    # 4. Perform image inpainting
    im_inpainted = get_image_inpainted(image=im, image_mask=im_mask)
    im_inpainted = cv2.cvtColor(im_inpainted, cv2.COLOR_BGR2RGB) # im_inpainted is in BGR format, convert to RGB

    # 5. Get classification =========================================== #
    # Process text and image for harmful/benign
    proba, label = 1, 0

    return proba, label


if __name__ == "__main__":
    file_paths = []

    with multiprocessing.Pool(4) as pool:
        ans = pool.map(process_line_by_line, ["./local_test/test_images/image108.png" for _ in range(100)])

    # Iteration loop to get new image filepath from sys.stdin:
    # for line in sys.stdin:
    #     # IMPORTANT: Please ensure any trailing whitespace (eg: \n) is removed. This may impact some modules to open the filepath
    #     image_path = line.rstrip()
    #     file_paths.append(image_path)

    # try:
    #     with multiprocessing.Pool(2) as pool:
    #         ans = pool.map(process_line_by_line, file_paths)

    #     # Ensure each result for each image_path is a new line
    #     for proba, label in ans:
    #         sys.stdout.write(f"{proba:.4f}\t{label}\n")

    # except Exception as e:
    #     # Output to any raised/caught error/exceptions to stderr
    #     sys.stderr.write(str(e))


