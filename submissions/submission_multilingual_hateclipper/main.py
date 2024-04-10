import easyocr
import sys

from hateclipper_dataset import CustomCollator, load_dataset
from hateclipper_engine import CLIPClassifier

from torch.utils.data import DataLoader
import numpy as np
import torch
import yaml
import random
import argparse
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# reader = easyocr.Reader(['en', 'ch_sim', 'ch_tra', 'ms', 'ta'])
reader = easyocr.Reader(['en'], model_storage_directory='./.EasyOCR/model', user_network_directory='./.EasyOCR/user_network')

def move_to_cuda(x):
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = move_to_cuda(v)
    elif isinstance(x, tuple):
        x = tuple(move_to_cuda(v) for v in x)
    else:
        x = x.to("cuda")
    return x


def main(args, texts, filepaths):
    dataset = load_dataset(args, texts, filepaths)

    num_cpus = min(args.batch_size, 6)
    collator = CustomCollator(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator)

    # create model
    model = CLIPClassifier(args).to("cuda")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cuda")["state_dict"])

    probas = []
    labels = []

    for batch in tqdm(dataloader):
        with torch.inference_mode():
            batch = move_to_cuda(batch)
            prob, pred = model(batch)
            probas += prob
            labels += pred

    return probas, labels


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        args = yaml.safe_load(f)

        if args["seed"] is not None:
            SEED = args['seed']
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            np.random.seed(SEED)
            random.seed(SEED)

    args = argparse.Namespace(**args)

    file_paths = []
    for line in sys.stdin:
        image_path = line.rstrip()
        file_paths.append(image_path)

    try:
        texts = []
        for fp in file_paths:
            text = reader.readtext(fp, detail=0)
            text = " ".join(text)
            texts.append(text)

        probas, labels = main(args, texts, file_paths)

        for i in range(len(labels)):
            proba, label = probas[i], labels[i]
            sys.stdout.write(f"{proba:.4f}\t{label}\n")

    except Exception as e:
        sys.stderr.write(str(e))
