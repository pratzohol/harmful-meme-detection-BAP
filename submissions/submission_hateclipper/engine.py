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

from transformers import CLIPModel, AutoConfig, AutoModel


class CLIPClassifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.dataset = args.dataset

        self.use_pretrained_map = args.use_pretrained_map
        self.num_mapping_layers = args.num_mapping_layers
        self.map_dim = args.map_dim
        self.num_pre_output_layers = args.num_pre_output_layers

        self.fusion = args.fusion
        self.lr = args.lr

        # decay and regularization params
        self.weight_decay = args.weight_decay

        self.acc = torchmetrics.Accuracy(task='binary')
        if self.dataset == 'prop':
            self.auroc = torchmetrics.AUROC(num_classes=22)
            self.precision_score = torchmetrics.Precision(mdmc_average='global')
            self.recall = torchmetrics.Recall(mdmc_average='global')
            self.f1 = torchmetrics.F1Score(mdmc_average='global')
        else:
            self.auroc = torchmetrics.AUROC(task='binary')
            self.precision_score = torchmetrics.Precision(task='binary')
            self.recall = torchmetrics.Recall(task='binary')
            self.f1 = torchmetrics.F1Score(task='binary')

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./")
        if args.image_encoder == 'clip':
            self.image_encoder = copy.deepcopy(self.clip.vision_model)
        else:
            raise ValueError()

        if args.text_encoder == 'clip':
            self.text_encoder = copy.deepcopy(self.clip.text_model)
        else:
            raise ValueError()

        if self.use_pretrained_map:
            self.image_map = nn.Sequential(
                copy.deepcopy(self.clip.visual_projection),
                nn.ReLU(),
                nn.Linear(self.clip.projection_dim, self.map_dim)
            )
            self.text_map = nn.Sequential(
                copy.deepcopy(self.clip.text_projection),
                nn.ReLU(),
                nn.Linear(self.clip.projection_dim, self.map_dim)
            )
        else:
            image_map_layers = [nn.Linear(self.image_encoder.config.hidden_size, self.map_dim), nn.Dropout(p=0.1)]
            text_map_layers = [nn.Linear(self.text_encoder.config.hidden_size, self.map_dim), nn.Dropout(p=0.1)]

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

        output_input_dim = self.map_dim
        if self.dataset in ['fb-meme', 'tamil']:
            self.output = nn.Linear(output_input_dim, 1)
        elif self.dataset == 'prop':
            self.output = nn.Linear(output_input_dim, 22)

        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        for _, p in self.image_encoder.named_parameters():
            p.requires_grad_(False)

        for _, p in self.text_encoder.named_parameters():
            p.requires_grad_(False)

    def forward(self, batch):
        image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
        image_features = self.image_map(image_features)

        if self.text_encoder_name == 'clip':
            text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
            text_features = self.text_map(text_features)
        else:
            raise ValueError()

        image_features = F.normalize(image_features, p=2, dim=1)  # [batch_size, d]
        text_features = F.normalize(text_features, p=2, dim=1)  # [batch_size, d]

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

        text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
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

        if self.dataset in ['tamil', 'prop']:
            output['precision'] = self.precision_score(preds, batch['labels'])
            output['recall'] = self.recall(preds, batch['labels'])
            output['f1'] = self.f1(preds, batch['labels'])

        return output

    def training_step(self, batch, batch_idx):
        output = self.common_step(batch)

        self.log('train/loss', output['loss'])
        self.log('train/accuracy', output['accuracy'])
        self.log('train/auroc', output['auroc'])

        if self.dataset in ['tamil', 'prop']:
            self.log('train/precision', output['precision'])
            self.log('train/recall', output['recall'])
            self.log('train/f1', output['f1'])

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

        if self.dataset in ['tamil', 'prop']:
            self.log('val/precision', output['precision'])
            self.log('val/recall', output['recall'])
            self.log('val/f1', output['f1'])

        return output['loss']

    def test_step(self, batch, batch_idx):
        output = self.common_step(batch)

        self.log(f'test/accuracy', output['accuracy'])
        self.log(f'test/auroc', output['auroc'])
        self.log(f'test/loss', output['loss'])

        if self.dataset in ['tamil', 'prop']:
            self.log(f'test/precision', output['precision'])
            self.log(f'test/recall', output['recall'])
            self.log(f'test/f1', output['f1'])

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
