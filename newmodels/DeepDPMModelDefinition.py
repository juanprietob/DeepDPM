#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import os
from typing import List, Any

import torch
import argparse
import torch.nn as nn
import pytorch_lightning as pl

from newmodels.DeepDPMFeatureExtractor import FeatureExtractor
from pytorch_lightning.loggers import NeptuneLogger
import numpy as np

from src.AE_ClusterPipeline import AE_ClusterPipeline
from src.datasets import MNIST, REUTERS
from src.embbeded_datasets import embbededDataset
from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel

from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from DeepDPM import cluster_acc


# This replaces the file AE_ClusterPipeline
class DeepDPMModelDefinition(pl.LightningModule):
    def __init__(self, logger, args):
        super(DeepDPMModelDefinition, self).__init__()
        self.args = args
        self.pretrain_logger = logger
        self.args = args
        self.beta = args.beta  # coefficient of the clustering term
        self.lambda_ = args.lambda_  # coefficient of the reconstruction term
        self.n_clusters = self.args.n_clusters
        if self.args.seed:
            pl.utilities.seed.seed_everything(self.args.seed)

        # Validation check
        if not self.beta > 0:
            msg = "beta should be greater than 0 but got value = {}."
            raise ValueError(msg.format(self.beta))

        if not self.lambda_ > 0:
            msg = "lambda should be greater than 0 but got value = {}."
            raise ValueError(msg.format(self.lambda_))

        if len(self.args.hidden_dims) == 0:
            raise ValueError("No hidden layer specified.")

        # Define feature extractor
        self.feature_extractor = FeatureExtractor(args)
        self.args.latent_dim = self.feature_extractor.latent_dim
        self.criterion = nn.MSELoss(reduction="sum")

        # Define cluster creator
        self.clustering = ClusterNet(args, self)
        self.init_clusternet_num = 0  # number of times the clustering net was initialized
        self.plot_utils = PlotUtils(hparams=self.args)

    def forward(self, x, latent=False):
        # Steps:
        # 1. Feature extractor for latent features. Encodes but does not decode.
        latent_X = self.feature_extractor(x, latent=True)

        # 2. If latent, return latentX. This is used by ClusterNet primarily.
        if latent:
            return latent_X.detach().cpu().numpy()

        # 2.5 Reduce from many dimensions to 2 dimensions.
        latent_X = torch.flatten(latent_X, start_dim=1)

        # 3. If not latent and not pseuodolabel, cluster update assign
        if self.args.cluster_assignments != "pseudo_label":
            return latent_X, self.clustering.update_assign(latent_X, self.args.cluster_assignments)
        # 4. Otherwise, return 0
        else:
            return 0


    def training_step(self, batch, batch_idx):
        pass

    def on_train_epoch_start(self) -> None:
        # for plotting
        self.sampled_codes = torch.empty(0)
        self.sampled_gt = torch.empty(0)
        if self.current_epoch == 0:
            self.log('data_stats/train_n_samples', len(self.train_dataloader().dataset))
            self.log('data_stats/val_n_samples', len(self.val_dataloader().dataset))
            if self.args.pretrain:
                assert self.args.pretrain_epochs > 0
                print("========== Start pretraining ==========")
                self.pretrain = True
            else:
                self.pretrain = False
                # using pretrained weights only initialize clusters
                assert self.args.pretrain_path is not None
                self._init_clusters()

        # Remove multiple lines for saving weights, as PL does this automatically after each epoch

    def validation_step(selfself, batch, batch_idx):
        pass

    def validation_epoch_end(self, validation_step_outputs: List[Any]) -> None:
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.wd
        )
        return optimizer
