# https://github.com/lkulowski/LSTM_encoder_decoder
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn

from models.encoders import RNNEncoder
from models.seq_autoencoder import SequenceAutoencoder
from datasets.SR_timeseries import SR_Timeseries, pad_collate
#from datasets.data_generator import ToyDataset, pad_collate

dataset_SR = SR_Timeseries('./data/patch_average_ts_300m.txt')
lr_init = 1E-2
num_epochs = 25
model_output = 'temporal_AE_model.pt'
code_size = 8

args = {}
#encoder parameters
args["encoder_arch"] = "RNN"
args['encoder_args'] = {}
args['encoder_args']['input_size'] = 4
args['encoder_args']['hidden_size'] = code_size
args['encoder_args']['layers'] = 1

#decoder parameters
args["decoder_arch"] = "RNN"
args['decoder_args'] = {}
args['decoder_args']['input_size'] = 4
args['decoder_args']['hidden_size'] = code_size
args['decoder_args']['layers'] = 1

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# dataset = ToyDataset(5, 15)
# eval_dataset = ToyDataset(5, 15, type='eval')
BATCHSIZE = 16
train_loader = data.DataLoader(dataset_SR, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate, drop_last=True)
eval_loader = data.DataLoader(dataset_SR, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate, drop_last=True)

model = SequenceAutoencoder(args)  #RNNEncoder(args)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.999), weight_decay=0)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95);
criterion = nn.MSELoss()

best_loss = float('inf')
for epoch in range(num_epochs):

    batch_loss = 0.0
    n_samples = 0
    for batch in train_loader:
        #run model
        x = batch['sr_data'].to(device)
        x_len = batch['lens']
        target_len = 16
        output = model(x, x_len, target_len)

        #optimize model
        optimizer.zero_grad()
        loss = criterion(output, x)  #autoencoder error
        loss.backward()  #backpropogate error
        optimizer.step() #update model parameters


        batch_loss += loss.item()
        n_samples += x.shape[0]


    #finished epoch
    lr_scheduler.step()
    epoch_avg_loss = batch_loss/n_samples
    print('epoch: {}, avg_loss: {}'.format(epoch, epoch_avg_loss))

    if epoch_avg_loss < best_loss:
        best_loss = epoch_avg_loss
        torch.save(model.state_dict(), model_output)

    #print('ran batch')

orig_data = open('original_data.txt', 'w')
ae_preds = open('autoencoder_pred.txt', 'w')

for batch in eval_loader:
    #run model
    x = batch['sr_data'].to(device)
    x_len = batch['lens']
    target_len = 16
    output = model(x, x_len, target_len)

    #write out data
    cpu = torch.device("cpu")
    x_np = x.to(cpu).numpy()
    output_np = output.to(cpu).detach().numpy()

    n_samples = x_np.shape[0]
    n_timepts = x_np.shape[1]
    n_channels = x_np.shape[2]

    for i in range(n_samples):
        for j in range(n_timepts):
            str1 = '\t'.join([str(x) for x in x_np[i, j, :]]) + "\t"
            orig_data.write(str1)

            str2 = '\t'.join([str(x) for x in output_np[i, j, :]]) + "\t"
            ae_preds.write(str2)

        orig_data.write('\n')
        ae_preds.write('\n')

orig_data.close()
ae_preds.close()

print('hello')


