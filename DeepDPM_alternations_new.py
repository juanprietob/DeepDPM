#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

from src.AE_ClusterPipeline import AE_ClusterPipeline
from src.datasets import MNIST, REUTERS
from src.embbeded_datasets import embbededDataset
from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel

from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from DeepDPM import cluster_acc

from newmodels.DeepDPMModelDefinition import DeepDPMModelDefinition
from newmodels.DeepDPMDataModule import DeepDPMDataModule

from typing import Union, List


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument("--dir", default="/path/to/dataset/", help="dataset directory")
    parser.add_argument("--dataset", default="mnist")

    # Training parameters
    parser.add_argument(
        "--lr", type=float, default=0.002, help="learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--wd", type=float, default=5e-4, help="weight decay (default: 5e-4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="input batch size for training"
    )
    parser.add_argument(
        "--epoch", type=int, default=100, help="number of epochs to train"
    )
    parser.add_argument(
        "--pretrain_epochs", type=int, default=0, help="number of pre-train epochs"
    )

    parser.add_argument(
        "--pretrain", action="store_true", help="whether use pre-training"
    )

    parser.add_argument(
        "--pretrain_path",
        type=str,
        default="./saved_models/ae_weights/mnist_e2e.zip",
        help="use pretrained weights",
    )

    # Add additional Model parameters, rather than adding them here?
    parser = DeepDPMModelDefinition.add_model_specific_args(parser)
    parser = ClusterNetModel.add_model_specific_args(parser)

    # Utility parameters
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="number of jobs to run in parallel"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device for computation (default: cpu)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=400,
        help="how many batches to wait before logging the training status",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="short test run on a few instances of the dataset",
    )

    # Logger parameters
    parser.add_argument(
        "--tag",
        type=str,
        default="Replicate git results",
        help="Experiment name and tag",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed",
    )

    parser.add_argument(
        "--features_dim",
        type=int,
        default=128,
        help="features dim of embedded datasets",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=300,
        help="number of AE epochs",
    )
    parser.add_argument(
        "--number_of_ae_alternations",
        type=int,
        default=3,
        help="The number of DeepDPM AE alternations to perform",
    )
    parser.add_argument("--save_checkpoints", type=bool, default=False)
    parser.add_argument("--exp_name", type=str, default="default_exp")
    parser.add_argument(
        "--offline", action="store_true", help="Run training without Neptune Logger"
    )
    parser.add_argument("--gpus", default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Steps:
    # 1. Parse arguments, set global defaults
    args = parse_args()
    args.n_clusters = args.init_k

    if args.seed:
        pl.utilities.seed.seed_everything(args.seed)

    # 2. Load data. Do through Lightning.
    famli = DeepDPMDataModule()

    # TODO: issue with args.input_dim being incorrect at Step 5.
    args.input_dim = args.features_dim

    # 3. Initialize loggers
    tags = ["DeepDPM with alternations"]
    tags.append(args.tag)
    logger = DummyLogger()

    # 4. Initialize checkpoints.
    if args.save_checkpoints:
        if not os.path.exists(f"./saved_models/{args.dataset}"):
            os.makedirs(f"./saved_models/{args.dataset}")
        os.makedirs(f"./saved_models/{args.dataset}/{args.exp_name}")
    checkpoint_callback = ModelCheckpoint(
        filename=f"deepdpm_",
        monitor="cluster_acc",
        save_top_k=10,
        mode="min",
    )

    # 5. Initialize model, load from checkpoint if pretraining.
    model = DeepDPMModelDefinition(args=args, logger=logger)

    # if args.pretrain:
    #     model = DeepDPMModelDefinition.load_from_checkpoint(args.pretrain_path)

    max_epochs = args.epoch * (args.number_of_ae_alternations - 1) + 1

    # 6. PyTorch lightning trainer initialize
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        gpus=args.gpus,
        num_sanity_val_steps=0,
        callbacks=checkpoint_callback,
    )
    trainer.fit(model, datamodule=famli)

    # 7. Test the model
    trainer.test(ckpt_path="best", datamodule=famli)
