from pyexpat import features
import copy
import math
from sys import prefix
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from multilingual_clip import pt_multilingual_clip
from transformers import CLIPModel, AutoConfig, AutoModel
from transformers import CLIPTokenizer, CLIPProcessor, AutoTokenizer

class CLIPClassifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.dataset = args.dataset
        self.args = args

        self.num_mapping_layers = args.num_mapping_layers
        self.map_dim = args.map_dim
        self.num_pre_output_layers = args.num_pre_output_layers

        self.fusion = args.fusion
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.acc = torchmetrics.Accuracy(task='binary')
        self.auroc = torchmetrics.AUROC(task='binary')
        self.precision_score = torchmetrics.Precision(task='binary')
        self.recall = torchmetrics.Recall(task='binary')
        self.f1 = torchmetrics.F1Score(task='binary')

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./")
        if args.image_encoder == 'clip':
            self.image_encoder = copy.deepcopy(self.clip.vision_model)
            image_map_layers = [nn.Linear(self.image_encoder.config.hidden_size, self.map_dim), nn.Dropout(p=0.1)]

        if args.text_encoder == 'clip':
            text_map_layers = [nn.Linear(self.clip.text_model.config.hidden_size, self.map_dim), nn.Dropout(p=0.1)]
        elif args.text_encoder == 'xlm-roberta-large':
            self.text_encoder = pt_multilingual_clip.MultilingualCLIP.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-L-14", cache_dir="./")
            text_map_layers = [nn.Linear(self.text_encoder.config.numDims, self.map_dim), nn.Dropout(p=0.1)]

        self.image_map = nn.Sequential(*image_map_layers)
        self.text_map = nn.Sequential(*text_map_layers)

        del self.clip

        if args.fusion == 'align':
            pre_output_input_dim = self.map_dim
        elif args.fusion == 'concat':
            pre_output_input_dim = self.map_dim*2
        elif args.fusion.startswith('cross'):
            pre_output_input_dim = self.map_dim**2
        else:
            raise ValueError()

        pre_output_layers = [nn.Dropout(p=0.4), nn.Linear(pre_output_input_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=0.2)]
        self.pre_output = nn.Sequential(*pre_output_layers)

        self.output = nn.Linear(self.map_dim, 1)
        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        for _, p in self.image_encoder.named_parameters():
            p.requires_grad_(False)

        for _, p in self.text_encoder.named_parameters():
            p.requires_grad_(False)

    def forward(self, batch):
        image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
        image_features = self.image_map(image_features)

        if self.args.text_encoder == 'clip':
            text_output = self.text_processor([txt for txt in batch["text"]], padding=True, return_tensors="pt", truncation=True)
            input_ids = text_output['input_ids']
            attention_mask = text_output['attention_mask']
            text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            text_features = self.text_map(text_features)
        elif self.args.text_encoder == 'xlm-roberta-large':
            text_features = self.text_encoder([txt for txt in batch["text"]], tokenizer=self.text_processor)
            text_features = self.text_map(text_features)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)


        if self.fusion == 'align':
            features = torch.mul(image_features, text_features)  # [batch_size, d]
        elif self.fusion == 'concat':
            features = torch.cat([image_features, text_features], dim=1) # [batch_size, 2*d]
        elif self.fusion.startswith('cross'):
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1))  # [batch_size, d, d]
            features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        else:
            raise ValueError()

        features = self.pre_output(features)
        logits = self.output(features)
        preds = (torch.sigmoid(logits) >= 0.5).long()

        return preds

    def common_step(self, batch):
        image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
        image_features = self.image_map(image_features)

        # if self.args.text_encoder == 'clip':
        #     text_output = self.text_processor([txt for txt in batch["text"]], padding=True, return_tensors="pt", truncation=True)
        #     input_ids = text_output['input_ids']
        #     attention_mask = text_output['attention_mask']
        #     text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_outpu
        # elif self.args.text_encoder == 'xlm-roberta-large':
        #     breakpoint()
        #     text_features = self.text_encoder([txt for txt in batch["text"]], tokenizer=self.text_processor)

        text_features = self.text_map(text_features)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        output = {}
        if self.fusion == 'align':
            features = torch.mul(image_features, text_features)
        elif self.fusion == 'concat':
            features = torch.cat([image_features, text_features], dim=1)
        elif self.fusion.startswith('cross'):
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1))  # [16, d, d]
            features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        else:
            raise ValueError()

        features_pre_output = self.pre_output(features)
        logits = self.output(features_pre_output).squeeze(dim=1)  # [batch_size, 1(or)n]
        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()

        output['loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
        output['accuracy'] = self.acc(preds, batch['labels'])
        output['auroc'] = self.auroc(preds_proxy, batch['labels'])

        return output

    def training_step(self, batch, batch_idx):
        output = self.common_step(batch)

        self.log('train/loss', output['loss'])
        self.log('train/accuracy', output['accuracy'])
        self.log('train/auroc', output['auroc'])
        return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch)

        self.log_dict(
            {
                'val/loss': output['loss'],
                'val/accuracy': output['accuracy'],
                'val/auroc': output['auroc']
            },
            on_epoch=True
        )
        return output['loss']

    def test_step(self, batch, batch_idx):
        output = self.common_step(batch)

        self.log(f'test/accuracy', output['accuracy'])
        self.log(f'test/auroc', output['auroc'])
        self.log(f'test/loss', output['loss'])
        return output

    def on_train_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def on_validation_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def on_test_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for _, p in self.named_parameters() if p.requires_grad]}
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
