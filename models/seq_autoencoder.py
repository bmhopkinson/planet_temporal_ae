import torch
from torch import nn

import models.encoders as encoders
import models.decoders as decoders


class SequenceAutoencoder(nn.Module):
    def __init__(self, args):
        super(SequenceAutoencoder, self).__init__()

        self.encoder = None
        self.decoder = None
        self.input_size = args['decoder_args']['input_size']

        if args["encoder_arch"] == "RNN":
            self.encoder = encoders.RNNEncoder(args['encoder_args'])
        else:
            raise NotImplementedError

        if args["decoder_arch"] == "RNN":
            self.decoder = decoders.RNNDecoder(args['decoder_args'])
        else:
            raise NotImplementedError

    def forward(self, input, input_len, target_len):

        #encoder
        output_encoder, (hidden, cell) = self.encoder(input, input_len)

        #decoder
        batch_size = input.size(0)
        output = torch.zeros(batch_size, 1, self.input_size)  #initialize output/next input
        output_ts = torch.zeros(batch_size, target_len, self.input_size)  #initialize output/next input

        for i in range(target_len):
            output, (hidden, cell) = self.decoder(output, hidden, cell)
            output_ts[:, i, :] = output.squeeze(1)

        return output_ts