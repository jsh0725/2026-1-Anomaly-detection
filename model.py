import torch
import torch.nn as nn


class LSTMAE(nn.Module):
    def __init__(self, n_features, hidden_size=64, latent_size=16, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.enc_fc = nn.Linear(hidden_size, latent_size)
        self.dec_fc = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=n_features, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len, features)
        enc_out, (h, c) = self.encoder(x)
        # take last time-step
        last = enc_out[:, -1, :]
        z = self.enc_fc(last)
        dec_in = self.dec_fc(z).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in)
        return dec_out
