from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, CLIPProcessor


class HatefulMemesDataset(Dataset):
    def __init__(self, texts, filepaths, image_size=224):
        super(HatefulMemesDataset, self).__init__()
        self.image_size = image_size
        self.texts = texts
        self.filepaths = filepaths

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        fp = self.filepaths[idx]

        item = {}
        item['image'] = Image.open(fp).convert('RGB').resize((self.image_size, self.image_size))
        item['text'] = txt
        return item


class CustomCollator(object):
    def __init__(self, args):
        self.args = args
        self.image_processor = CLIPProcessor.from_pretrained("models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41")
        self.text_processor = CLIPTokenizer.from_pretrained("models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41")

    def __call__(self, batch):
        pixel_values = self.image_processor(images=[item['image'] for item in batch], return_tensors="pt")['pixel_values']
        text_output = self.text_processor([item['text'] for item in batch], padding=True, return_tensors="pt", truncation=True)

        batch_new = {}
        batch_new['pixel_values'] = pixel_values,
        batch_new['input_ids'] = text_output['input_ids']
        batch_new['attention_mask'] = text_output['attention_mask']

        return batch_new


def load_dataset(args, texts, filepaths):
    dataset = HatefulMemesDataset(texts, filepaths, image_size=args.clip_image_size)
    return dataset
