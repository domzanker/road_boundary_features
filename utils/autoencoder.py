import pytorch_lightning as pl
from utils.feature_net import FeatureNet, Encoder, Decoder
from typing import List, Union
from torch.nn import Module, Identity, ConvTranspose2d
from torch.common_types import _size_2_t


class AutoEncoder(FeatureNet):
    def __init__(self, configs):
        super(AutoEncoder, self).__init__(configs)
        self.encoder_configs = configs["model"]["encoder"]
        self.decoder_configs = configs["model"]["decoder"]

        self.encoder_prec = Identity()
        self.decoder = AEDecoder(**self.decoder_configs)
        self.head = Identity()


class AEDecoder(Decoder):
    def __init__(self, *args, **kwargs):
        super(AEDecoder, self).__init__(*args, **kwargs)

    def forward(self, *features):
        # features = features[1:]
        # [print(f.shape) for f in features]
        features = features[::-1]
        x = features[0]
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x
