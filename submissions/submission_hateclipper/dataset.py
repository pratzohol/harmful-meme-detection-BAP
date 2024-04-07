import os
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, CLIPProcessor, AutoTokenizer


class FBHatefulMemesDataset(Dataset):
    def __init__(self, root_folder, split='train', image_size=224, text='original'):
        super(FBHatefulMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.split = split
        self.image_size = image_size
        self.text = text

        self.info_file = os.path.join(root_folder, 'fb_hateful_memes_info.csv')
        self.df = pd.read_csv(self.info_file)
        self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)
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

        item['label'] = row['label']
        item['idx_meme'] = row['id']
        return item


class TamilMemesDataset(Dataset):
    def __init__(self, root_folder, split='train', image_size=224):
        """
        First, preprocess Tamil Troll Memes using `hateclipper/preprocessing/format_tamil_memes.ipynb`
        """
        super(TamilMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.split = split
        self.image_size = image_size
        self.info_file = os.path.join(root_folder, 'labels.csv')
        self.df = pd.read_csv(self.info_file)
        self.df = self.df[self.df['split'] ==
                          self.split].reset_index(drop=True)
        self.fine_grained_labels = []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        item['image'] = Image.open(f"{self.root_folder}/{row['meme_path']}").convert(
            'RGB').resize((self.image_size, self.image_size))
        item['text'] = row['text']
        # named as caption just to match the format of FBHatefulMemesDataset
        item['caption'] = row['text_transliterated']
        item['label'] = row['is_troll']

        return item


class PropMemesDataset(Dataset):
    def __init__(self, root_folder, split='train', image_size=224):
        super(PropMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.split = split
        self.image_size = image_size
        self.info_file = os.path.join(
            root_folder, f'annotations/{self.split}.jsonl')
        self.df = pd.read_json(self.info_file, lines=True)
        self.fine_grained_labels = ['Black-and-white Fallacy/Dictatorship', 'Name calling/Labeling', 'Smears', 'Reductio ad hitlerum', 'Transfer', 'Appeal to fear/prejudice',
                                    'Loaded Language', 'Slogans', 'Causal Oversimplification', 'Glittering generalities (Virtue)', 'Flag-waving', "Misrepresentation of Someone's Position (Straw Man)",
                                    'Exaggeration/Minimisation', 'Repetition', 'Appeal to (Strong) Emotions', 'Doubt', 'Obfuscation, Intentional vagueness, Confusion', 'Whataboutism', 'Thought-terminating cliché',
                                    'Presenting Irrelevant Data (Red Herring)', 'Appeal to authority', 'Bandwagon']
        mlb = MultiLabelBinarizer().fit([self.fine_grained_labels])
        self.df = self.df.join(pd.DataFrame(mlb.transform(self.df['labels']),
                                            columns=mlb.classes_,
                                            index=self.df.index))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        item['image'] = Image.open(f"{self.root_folder}/images/{row['image']}").convert(
            'RGB').resize((self.image_size, self.image_size))
        item['text'] = " ".join(row['text'].replace(
            "\n", " ").strip().lower().split())
        item['labels'] = row[self.fine_grained_labels].values.tolist()
        for label in self.fine_grained_labels:
            item[label] = row[label]

        return item


class CustomCollator(object):
    def __init__(self, args):
        self.args = args
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./")
        self.text_processor = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./")

    def __call__(self, batch):
        pixel_values = self.image_processor(images=[item['image'] for item in batch], return_tensors="pt")['pixel_values']
        text_output = self.text_processor([item['text'] for item in batch], padding=True, return_tensors="pt", truncation=True)

        if self.args.dataset in ['fb-meme', 'tamil']:
            labels = torch.LongTensor([item['label'] for item in batch])

        if self.args.dataset in ['fb-meme']:
            idx_memes = torch.LongTensor([item['idx_meme'] for item in batch])

        batch_new = {}
        batch_new['pixel_values'] = pixel_values,
        batch_new['input_ids'] = text_output['input_ids']
        batch_new['attention_mask'] = text_output['attention_mask']

        if self.args.dataset in ['fb-meme', 'tamil']:
            batch_new['labels'] = labels
        if self.args.dataset in ['fb-meme']:
            batch_new['idx_memes'] = idx_memes
        if self.args.dataset == 'prop':
            batch_new['labels'] = torch.LongTensor([item['labels'] for item in batch])

        return batch_new


def load_dataset(args, split):
    root_folder = "/home/pratzohol/google-drive/work-stuff/harmful-meme-detection/datasets/fb-meme"
    if args.dataset == 'tamil':
        dataset = TamilMemesDataset(root_folder=root_folder, split=split, image_size=args.clip_image_size)
    elif args.dataset == 'prop':
        dataset = PropMemesDataset(root_folder=root_folder, split=split, image_size=args.clip_image_size)
    else:
        dataset = FBHatefulMemesDataset(root_folder=root_folder, split=split, image_size=args.clip_image_size, text=args.text)
    return dataset
