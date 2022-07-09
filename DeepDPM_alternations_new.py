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
        "--pretrain_path", type=str, default="./saved_models/ae_weights/mnist_e2e", help="use pretrained weights"
    )

    # Model parameters
    parser = AE_ClusterPipeline.add_model_specific_args(parser)
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
        help="The number of DeepDPM AE alternations to perform"
    )
    parser.add_argument(
        "--save_checkpoints", type=bool, default=False
    )
    parser.add_argument(
        "--exp_name", type=str, default="default_exp"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run training without Neptune Logger"
    )
    parser.add_argument(
        "--gpus",
        default=None
    )
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
    if args.dataset == "mnist":
        data = MNIST(args)
    elif args.dataset == "reuters10k":
        data = REUTERS(args, how_many=10000)
    else:
        # Used for ImageNet-50
        data = embbededDataset(args)

    train_loader, val_loader = data.get_loaders()
    args.input_dim = data.input_dim

    # TODO: what we should have
    famli = DeepDPMDataModule()

    # 3. Initialize loggers
    tags = ['DeepDPM with alternations']
    tags.append(args.tag)
    logger = DummyLogger()

    device = "cuda" if torch.cuda.is_available() and args.gpus is not None else "cpu"

    # 4. Initialize checkpoints.
    if args.save_checkpoints:
        if not os.path.exists(f'./saved_models/{args.dataset}'):
            os.makedirs(f'./saved_models/{args.dataset}')
        os.makedirs(f'./saved_models/{args.dataset}/{args.exp_name}')
    checkpoint_callback = ModelCheckpoint(filename=f"deepdpm_e{epoch:03d}_vl{val_loss:.3f}",
                                          monitor="val_loss", save_top_k=10, mode="min")

    # 5. Initialize model, load from checkpoint if pretraining.
    model = DeepDPMModelDefinition(args=args, logger=logger)

    if not args.pretrain:
        model = DeepDPMModelDefinition.load_from_checkpoint(args.pretrain_path)


    max_epochs = args.epoch * (args.number_of_ae_alternations - 1) + 1

    # 6. PyTorch lightning trainer initialize
    trainer = pl.Trainer(logger=logger,
                         max_epochs=max_epochs,
                         gpus=args.gpus,
                         num_sanity_val_steps=0,
                         callbacks=checkpoint_callback,
                         )
    trainer.fit(model, datamodule=famli)


    # 7. Below should be handles under the validation/testing steps, rather than out here.
    model.to(device=device)
    DeepDPM = model.clustering.model.cluster_model
    DeepDPM.to(device=device)
    # evaluate last model
    for i, dataset in enumerate([data.get_train_data(), data.get_test_data()]):
        data_, labels_ = dataset.tensors[0], dataset.tensors[1].numpy()
        pred = DeepDPM(data_.to(device=device)).argmax(axis=1).cpu().numpy()

        acc = np.round(cluster_acc(labels_, pred), 5)
        nmi = np.round(NMI(pred, labels_), 5)
        ari = np.round(ARI(pred, labels_), 5)
        if i == 0:
            print("Train evaluation:")
        else:
            print("Validation evaluation")
        print(f"NMI: {nmi}, ARI: {ari}, acc: {acc}, final K: {len(np.unique(pred))}")
    model.cpu()
    DeepDPM.cpu()
