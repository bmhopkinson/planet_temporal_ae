# https://github.com/lkulowski/LSTM_encoder_decoder
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from torch.utils import data

from models.encoders import RNNEncoder
from models.seq_autoencoder import SequenceAutoencoder
from datasets.SR_timeseries import SR_Timeseries, pad_collate
#from datasets.data_generator import ToyDataset, pad_collate

dataset_SR = SR_Timeseries('./data/patch_average_ts_300m.txt')

args = {}


#encoder parameters
args["encoder_arch"] = "RNN"
args['encoder_args'] = {}
args['encoder_args']['input_size'] = 4
args['encoder_args']['hidden_size'] = 64
args['encoder_args']['layers'] = 1

#decoder parameters
args["decoder_arch"] = "RNN"
args['decoder_args'] = {}
args['decoder_args']['input_size'] = 4
args['decoder_args']['hidden_size'] = 64
args['decoder_args']['layers'] = 1


# dataset = ToyDataset(5, 15)
# eval_dataset = ToyDataset(5, 15, type='eval')
BATCHSIZE = 2
train_loader = data.DataLoader(dataset_SR, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate, drop_last=True)
# eval_loader = data.DataLoader(eval_dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate,
#                               drop_last=True)

model = SequenceAutoencoder(args)  #RNNEncoder(args)


for batch in train_loader:
    # x, y, x_len, y_len = sample_batch
    #
    x = batch['sr_data']
    x_len = batch['lens']
    target_len = 8
    output = model(x, x_len, target_len)
    #print('ran batch')

print('hello')


