import torch
import torch.nn as nn


class LSTMAE(nn.Module):
    def __init__(self, n_features, hidden_size=64, latent_size=16, num_layers=1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # 인코더
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True)
        self.enc_fc = nn.Linear(hidden_size, latent_size)
        
        # 디코더
        self.dec_fc = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, 
                                    num_layers=num_layers, batch_first=True)
        self.dec_out = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 인코더
        enc_out, (h, c) = self.encoder(x)
        last = enc_out[:, -1, :]  # (batch, hidden_size)
        z = self.enc_fc(last)      # (batch, latent_size)
        
        # 디코더
        dec_h = self.dec_fc(z)     # (batch, hidden_size)
        dec_in = dec_h.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_size)
        
        dec_out, _ = self.decoder_lstm(dec_in)  # (batch, seq_len, hidden_size)
        recon = self.dec_out(dec_out)           # (batch, seq_len, n_features)
        
        return recon

