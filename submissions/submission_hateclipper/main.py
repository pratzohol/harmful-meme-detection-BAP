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
reader = easyocr.Reader(['ch_tra', 'en'], model_storage_directory='./.EasyOCR/model', user_network_directory='./.EasyOCR/user_network')

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

    collator = CustomCollator(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

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

    # file_paths = [
    #     "./local_test/test_images/image108.png",
    #     "./local_test/test_images/image2248.png",
    #     "./local_test/test_images/image2259.png",
    #     "./local_test/test_images/image286.png",
    #     "./local_test/test_images/image124.png",
    #     "./local_test/test_images/image80.png",
    #     "./local_test/test_images/image194.png",
    #     "./local_test/test_images/image2214.png",
    #     "./local_test/test_images/image2201.png",
    #     "./local_test/test_images/image2202.png",
    #     "./local_test/test_images/image2210.png",
    #     "./local_test/test_images/image2204.png",
    #     "./local_test/test_images/image2209.png",
    #     "./local_test/test_images/image31.png",
    #     "./local_test/test_images/image2219.png",
    #     "./local_test/test_images/image276.png",
    #     "./local_test/test_images/image57.png",
    #     "./local_test/test_images/image2241.png",
    #     "./local_test/test_images/image69.png",
    #     "./local_test/test_images/image312.png"
    # ]
    # file_paths = file_paths * 30

    file_paths = []
    for line in sys.stdin:
        image_path = line.rstrip()
        file_paths.append(image_path)

    # import pandas as pd
    # df = pd.read_csv("../../datasets/fb-meme/fb_hateful_memes_info.csv")
    # root_folder = "../../datasets/fb-meme/"
    # indices = df["split"] == 'val'
    # file_paths = (root_folder + df["img"][indices]).to_list()
    # true_labels = df["label"][indices].astype(int).to_list()

    try:
        texts = []
        for fp in tqdm(file_paths):
            text = reader.readtext(fp, detail=0)
            text = " ".join(text)
            texts.append(text)

        probas, labels = main(args, texts, file_paths)

        # import torchmetrics
        # acc = torchmetrics.Accuracy(task='binary')
        # auroc = torchmetrics.AUROC(task='binary')

        # accuracy = acc(torch.tensor(labels), torch.tensor(true_labels))
        # auroc_score = auroc(torch.tensor(probas), torch.tensor(true_labels))

        # print(f"Accuracy: {accuracy:.4f}")
        # print(f"AUROC: {auroc_score:.4f}")

        for i in range(len(labels)):
            proba, label = probas[i], labels[i]
            sys.stdout.write(f"{proba:.4f}\t{label}\n")

    except Exception as e:
        sys.stderr.write(str(e))
