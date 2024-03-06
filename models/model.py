import random
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from utils.utility import *
from lightning.pytorch import LightningModule
from torchvision.utils import save_image
from torchvision.models import mobilenet_v3_large
from torch.optim.lr_scheduler import CosineAnnealingLR


class SimilarityModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model = mobilenet_v3_large()
        self.backbone = nn.Sequential(*list(model.children())[:-1])

        self.criterion = nn.TripletMarginLoss()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    @torch.no_grad()
    def forward(self, imgs):
        outputs = self.backbone(imgs).squeeze()

        return outputs

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch

        anchor_feature = self.backbone(anchor).squeeze()
        positive_feature = self.backbone(positive).squeeze()
        negative_feature = self.backbone(negative).squeeze()

        if self.config['loss'] == "cos":
            pos_sim = self.cos(anchor_feature, positive_feature)
            neg_sim = self.cos(anchor_feature, negative_feature)
            loss = torch.clamp(1 - pos_sim + neg_sim, min=0).mean()
            self.log('pos_sim', pos_sim.mean(), sync_dist=True, prog_bar=True, on_step=True)
            self.log('neg_sim', neg_sim.mean(), sync_dist=True, prog_bar=True, on_step=True)
        else: # Triplet Loss
            loss = self.criterion(anchor_feature, positive_feature, negative_feature)

        self.log('train_loss', loss, sync_dist=True, prog_bar=True, on_step=True)

        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, on_step=True)

        return {'loss' : loss}
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor_feature = self.backbone(anchor).squeeze()
        positive_feature = self.backbone(positive).squeeze()
        negative_feature = self.backbone(negative).squeeze()

        if self.config['loss'] == "cos":
            pos_sim = self.cos(anchor_feature, positive_feature)
            neg_sim = self.cos(anchor_feature, negative_feature)
            loss = torch.clamp(1 - pos_sim + neg_sim, min=0).mean()
        else:
            loss = self.criterion(anchor_feature, positive_feature, negative_feature)

        self.log('val_loss', loss, sync_dist=True, prog_bar=True, on_step=True)

        if self.config['save_image'] and self.trainer.global_rank == 0 and batch_idx < 2:
            random_indices = random.sample(range(len(anchor_feature)), 6)
            similarity = get_CosineSimilarity(anchor_feature)
            grid = get_SampleImage(similarity, anchor, random_indices)
            root_dir = Path(self.config['output_dir']) / 'images'
            root_dir.mkdir(exist_ok=True, parents=True)
            file_path = f'{root_dir}/sample_images_{self.current_epoch}_{batch_idx}.png'
            save_image(grid, file_path)

        return loss


    def configure_optimizers(self):
        optimizer = getattr(optim, self.config['OPTIM']['TYPE'])
        optimizer = optimizer(self.parameters(), lr=self.config['OPTIM']['LR'])

        T_max = self.config['TRAIN']['EPOCHS']
        eta_min = self.config['OPTIM']['LR'] * 0.01
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                'scheduler' : scheduler
            }
        }