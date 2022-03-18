import torch
from torch import nn

class RNNEncoder(nn.Module):

    def __init__(self, args):
        super(RNNEncoder, self).__init__()
        self.input_size = args['input_size']
        self.hidden_size = args['hidden_size']
        self.n_layers = args['layers']

        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.n_layers,
                           batch_first=True)


    def forward(self, input, input_lens):
        batch_size = input.shape[0]

        if input.is_cuda:
            device = input.get_device()
        else:
            device = torch.device('cpu')

        input_packed = torch.nn.utils.rnn.pack_padded_sequence(input, input_lens.to(torch.device("cpu")),
                                                                     batch_first=True, enforce_sorted=False)

        hidden_init, cell_init = (torch.randn(self.n_layers, batch_size, self.hidden_size).to(device),    #initialize hidden and cell state
                                  torch.randn(self.n_layers, batch_size, self.hidden_size).to(device))

        output, (hidden, cell) = self.rnn(input_packed, (hidden_init, cell_init))

        return output, (hidden, cell)






