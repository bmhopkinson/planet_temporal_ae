import torch
from torch import nn

import encoders
import decoders


class SequenceAutoencoder(nn.Module):
    def __init__(self, args):
        super(SequenceAutoencoder, self).__init__()

        self.encoder = None
        self.decoder = None

        if args["encoder_arch"] == "RNN":
            self.encoder = encoders.RNNEncoder(args['encoder_args'])
        else:
            raise NotImplementedError

        if args["decoder_arch"] == "RNN":
            self.decoder = decoders.RNN(args['decoder_args'])
        else:
            raise NotImplementedError

    def forward(self, input):
        code = self.encoder(input)
        output = self.decoder(code)
        return output