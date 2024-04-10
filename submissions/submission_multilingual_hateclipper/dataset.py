import os
import pandas as pd
import torch
import torch.nn as nn
import copy
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from multilingual_clip import pt_multilingual_clip
from transformers import CLIPTokenizer, CLIPProcessor, AutoTokenizer


class HatefulMemesDataset(Dataset):
    def __init__(self, dataset, root_folder, split='train', image_size=224, text='original'):
        super(HatefulMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.split = split
        self.image_size = image_size
        self.text = text

        if dataset == 'fb-meme':
            self.info_file = os.path.join(root_folder, 'fb_hateful_memes_multilingo_info.csv')
        elif dataset == 'rmmhs':
            self.info_file = os.path.join(root_folder, 'RMMHS_info.csv')

        self.df = pd.read_csv(self.info_file)

        if split == 'train':
            self.df = self.df[self.df['split'] == "train"].reset_index(drop=True)
        else:
            # import numpy as np
            # idx = np.arange(2500)
            # idxs = np.random.choice(idx,size=500, replace=False).tolist()
            # self.df = self.df[self.df['split'] == "val"].iloc[idxs].reset_index(drop=True)
            self.df = self.df[self.df['split'] == "val"].reset_index(drop=True)

            if dataset == 'rmmhs':
                self.df = self.df.tail(128)
        self.df["label"] = self.df["label"].fillna(-1).astype('int')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        item['image'] = Image.open(f"{self.root_folder}/{row['img']}").convert('RGB').resize((self.image_size, self.image_size))

        if self.text == 'original':
            item['text'] = row['text']
        elif self.text == 'easyocr':
            item['text'] = row['text_easyocr']

        if self.split == "test":
            item['text'] = row['text_easyocr']

        item['label'] = row['label']
        return item

class CustomCollator(object):
    def __init__(self, args):
        self.args = args
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./")

        if args.text_encoder == 'clip':
            self.text_processor = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./")
            self.text_encoder = copy.deepcopy(self.clip.text_model)
        elif args.text_encoder == 'xlm-roberta-large':
            self.text_processor = AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14', cache_dir="./")
            self.text_encoder = pt_multilingual_clip.MultilingualCLIP.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-L-14", cache_dir="./")
            text_map_layers = [nn.Linear(self.text_encoder.config.numDims, self.map_dim), nn.Dropout(p=0.1)]

    def __call__(self, batch):
        pixel_values = self.image_processor(images=[item['image'] for item in batch], return_tensors="pt")['pixel_values']

        if self.args.dataset in ['fb-meme', 'rmmhs']:
            labels = torch.LongTensor([item['label'] for item in batch])

        batch_new = {}
        batch_new['text'] = [item['text'] for item in batch]
        batch_new['pixel_values'] = pixel_values,

        if self.args.dataset in ['fb-meme', 'rmmhs']:
            batch_new['labels'] = labels

        return batch_new


def load_dataset(args, split):
    root_folder = args.data_path
    dataset = HatefulMemesDataset(dataset=args.dataset, root_folder=root_folder, split=split, image_size=args.clip_image_size, text=args.text)
    return dataset
