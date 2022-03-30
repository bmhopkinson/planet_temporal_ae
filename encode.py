import torch
from torch.utils import data
import numpy as np
import pandas as pd

from models.seq_autoencoder import SequenceAutoencoder
from datasets.SR_timeseries import SR_Timeseries, pad_collate

dataset_SR = SR_Timeseries('./data/patch_average_ts_30m.txt')
outfile = 'encoded_30m_data.txt'

model_path = 'temporal_AE_model.pt'
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

BATCHSIZE = 16
data_loader = data.DataLoader(dataset_SR, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate,
                               drop_last=False)
n_datapoints = len(dataset_SR)

model = SequenceAutoencoder(args)  # RNNEncoder(args)
model.eval()
model.to(device)

results = {'pos': pd.DataFrame({'lat':[], 'lon':[]}), 'code': np.zeros((n_datapoints, code_size))}
i_start = 0
i_end = 0
with torch.no_grad():
    for batch in data_loader:
        # extract data
        pos = batch['pos']
        x = batch['sr_data'].to(device)
        x_len = batch['lens']

        output = model.encode(x, x_len)  #run encoder and extract code
        code = output.to(torch.device("cpu")).squeeze().numpy()

        #record results
        results['pos'] = results['pos'].append(pos)

        batch_size = x.size(0)
        i_end += batch_size
        results['code'][i_start:i_end, :] = code

        i_start = i_end


results_np = np.hstack((results['pos'].to_numpy(), results['code']))
np.savetxt(outfile, results_np)
print('hello')
