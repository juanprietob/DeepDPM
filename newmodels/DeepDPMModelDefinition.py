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
from src.clustering_models.clusternet import ClusterNet

from src.clustering_models.clusternet_modules.utils.training_utils import training_utils
from src.clustering_models.clusternet_modules.utils.plotting_utils import PlotUtils

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
        self.init_clusternet_num = (
            0  # number of times the clustering net was initialized
        )

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
            return latent_X, self.clustering.update_assign(
                latent_X, self.args.cluster_assignments
            )
        # 4. Otherwise, return 0
        else:
            return 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            if self.feature_extractor is not None:
                codes = torch.from_numpy(
                    self.feature_extractor(x.view(x.size()[0], -1), latent=True)
                )
            else:
                codes = x

        if self.current_training_stage == "gather_codes":
            return self.only_gather_codes(codes, y, optimizer_idx)

        elif self.current_training_stage == "train_cluster_net":
            return self.cluster_net_pretraining(
                codes, y, optimizer_idx, x if batch_idx == 0 else None
            )

        else:
            raise NotImplementedError()

    def on_train_epoch_start(self) -> None:
        # for plotting
        self.sampled_codes = torch.empty(0)
        self.sampled_gt = torch.empty(0)
        if self.current_epoch == 0:
            self.log("data_stats/train_n_samples", len(self.train_dataloader().dataset))
            self.log("data_stats/val_n_samples", len(self.val_dataloader().dataset))
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
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Validation step")

    def validation_epoch_end(self, validation_step_outputs: List[Any]) -> None:
        raise NotImplementedError("Validation epoch end")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        data, labels = batch

        # PL automatically uses best model as choice.
        net_pred = self(data).argmax(axis=1).cpu().numpy()

        accuracies = {
            "cluster_acc": np.round(cluster_acc(labels, net_pred), 5),
            "nmi": np.round(NMI(net_pred, labels), 5),
            "ari": np.round(ARI(net_pred, labels), 5),
        }

        return accuracies

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.wd
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--lambda_",
            type=float,
            default=0.005,
            help="coefficient of the reconstruction loss",
        )
        parser.add_argument(
            "--beta",
            type=float,
            default=1,
            help="coefficient of the regularization term on " "clustering",
        )
        parser.add_argument(
            "--hidden-dims",
            type=int,
            nargs="+",
            default=[500, 500, 2000],
            help="hidden AE dims",
        )
        parser.add_argument(
            "--latent_dim", type=int, default=10, help="latent space dimension"
        )
        parser.add_argument(
            "--n-clusters",
            type=int,
            default=10,
            help="number of clusters in the latent space",
        )
        parser.add_argument(
            "--pretrain_noise_factor",
            type=float,
            default=0,
            help="the noise factor to be used in pretraining",
        )
        parser.add_argument(
            "--clustering",
            type=str,
            default="cluster_net",
            help="choose a clustering method",
        )
        parser.add_argument("--alternate", action="store_true")
        parser.add_argument(
            "--retrain_cluster_net_every",
            type=int,
            default=100,
        )
        parser.add_argument("--init_cluster_net_using_centers", action="store_true")
        parser.add_argument("--reinit_net_at_alternation", action="store_true")
        parser.add_argument(
            "--regularization",
            type=str,
            choices=["dist_loss", "cluster_loss"],
            help="which cluster regularization to use on the AE",
            default="dist_loss",
        )
        parser.add_argument(
            "--cluster_assignments",
            type=str,
            help="how to get the cluster assignment while training the AE, min_dist (hard assignment), forward_pass (soft assignment), pseudo_label (hard/soft assignment, TBD)",
            choices=["min_dist", "forward_pass", "pseudo_label"],
            default="min_dist",
        )
        parser.add_argument(
            "--update_clusters_params",
            type=str,
            choices=["False", "only_centers", "all_params", "all_params_w_prior"],
            default="False",
            help="whether and how to update the clusters params (e.g., center) during the AE training",
        )
        return parser
