import copy
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class CLIPClassifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.output = []

        self.num_mapping_layers = args.num_mapping_layers
        self.map_dim = args.map_dim
        self.num_pre_output_layers = args.num_pre_output_layers

        self.fusion = args.fusion
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./")
        if args.image_encoder == 'clip':
            self.image_encoder = copy.deepcopy(self.clip.vision_model)

        if args.text_encoder == 'clip':
            self.text_encoder = copy.deepcopy(self.clip.text_model)

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

        self.output = nn.Linear(self.map_dim, 1)
        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        for _, p in self.image_encoder.named_parameters():
            p.requires_grad_(False)

        for _, p in self.text_encoder.named_parameters():
            p.requires_grad_(False)

    def forward(self, batch):
        image_features = self.image_encoder(pixel_values=batch['pixel_values'][0]).pooler_output
        image_features = self.image_map(image_features)

        text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)

        image_features = F.normalize(image_features, p=2, dim=1)  # [batch_size, d]
        text_features = F.normalize(text_features, p=2, dim=1)  # [batch_size, d]

        if self.fusion == 'align':
            features = torch.mul(image_features, text_features)  # [batch_size, d]
        elif self.fusion == 'concat':
            features = torch.cat([image_features, text_features], dim=1)
        elif self.fusion.startswith('cross'):
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1))  # [batch_size, d, d]
            features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        else:
            raise ValueError()

        features = self.pre_output(features)

        logits = self.output(features)
        probas = torch.sigmoid(logits)
        preds = (probas >= 0.5).long()

        return probas.flatten(), preds.flatten()

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for _, p in self.named_parameters() if p.requires_grad]}
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
