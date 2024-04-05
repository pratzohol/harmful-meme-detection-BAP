# Standard Python Library
import sys
import multiprocessing

import cv2
import pytesseract
from pytesseract import Output
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

import torch
from peft import PeftModel

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

adapter_name = 'adapter_weights'
model_name = "original_llama"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map={"": 0},
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = PeftModel.from_pretrained(model, adapter_name)
model = model.merge_and_unload()
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1)

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
    return txt


def classifier(text):
    # implement here
    prompt = "I have given you some text in Singlish which is a mixture of English, Tamil, Malay and Chinese. You task is to predict whether the text is harmful (1) or benign (0). Text: " +text+' Response: '
    response = pipe(prompt)[0]['generated_text']
    try:
        label = int(response[len(prompt)])
    except:
        label = 1

    if label == 0:
        proba = 0.25
    else:
        proba = 0.85
    return proba, label


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
            proba, label = classifier(txt)
            sys.stdout.write(f"{proba:.4f}\t{label}\n")

    except Exception as e:
        # Output to any raised/caught error/exceptions to stderr
        sys.stderr.write(str(e))
