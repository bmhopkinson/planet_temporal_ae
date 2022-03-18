import torch
import torch.nn as nn

class RNNDecoder(nn.Module):
    def __init__(self, args):
        super(RNNDecoder, self).__init__()

        self.hidden_size = args['hidden_size']
        self.n_layers = args['layers']
        self.input_size = args['input_size']

        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.n_layers,
                           batch_first=True)

        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input, hidden, cell):
        output_rnn, (hidden, cell) = self.rnn(input, (hidden, cell))
        output = self.linear(output_rnn)

        return output, (hidden, cell)
