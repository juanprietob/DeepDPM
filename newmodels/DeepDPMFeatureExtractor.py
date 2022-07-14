import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .DeepDPMAutoEncoder import AutoEncoder, ConvAutoEncoder

from .DeepDPMContrastiveModel import ContrastiveModel


class FeatureExtractor(nn.Module):
    def __init__(self, args, feature_extractor=False, AE=True):
        super(FeatureExtractor, self).__init__()
        self.args = args
        self.feature_extractor = None
        self.latent_dim = None

        # Handle getting latent dimensions, so forth.
        if feature_extractor:
            self.feature_extractor = self.get_fe_model()
            self.latent_dim = self.feature_extractor.features_dim
        if AE:
            if self.args.dataset == "usps":
                self.autoencoder = ConvAutoEncoder(
                    self.args, input_dim=self.latent_dim or args.input_dim
                )
            else:
                self.autoencoder = AutoEncoder(
                    self.args, input_dim=self.latent_dim or args.input_dim
                )
            self.latent_dim = self.autoencoder.latent_dim

    def forward(self, X, latent=False):
        if self.feature_extractor:
            X = self.feature_extractor(X)
        if self.autoencoder:
            output = self.autoencoder.encoder(X)
            if latent:
                return output
            return self.autoencoder.decoder(output)
        return X

    def decode(self, latent_X):
        return self.autoencoder.decoder(latent_X)

    def extract_features(self, x):
        return self.feature_extractor(x)

    def get_fe_model(self, output_dim=128):
        backbone = self._get_backbone()
        model = ContrastiveModel(backbone=backbone, features_dim=output_dim)

        # Load pretrained weights
        if self.args.pretrain_path is not None and os.path.exists(
            self.args.pretrain_path
        ):
            state = torch.load(self.args.pretrain_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print("Loaded pretrained weights")
        return model

    # TODO: rewrite handling backbone
    def _get_backbone(self):
        if self.args.dataset in ("cifar-10", "cifar-20"):
            from src.feature_extractors.resnet_cifar import resnet18

            backbone = resnet18()
        elif self.args.dataset == "stl-10":
            from src.feature_extractors.resnet_stl import resnet18

            backbone = resnet18()
        elif "imagenet" in self.args.dataset:
            from src.feature_extractors.resnet import resnet50

            backbone = resnet50()
        return backbone
