import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MorphVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, num_filters,  device='cuda'):
        super(MorphVAE, self).__init__()

        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []

        for dim in hidden_dims:
            encoder_layers.append(nn.Conv3d(input_dim, outchannels=dim, kernel_size=3, stride=1, padding=1)) #should we apply padding
            encoder_layers.append(nn.BatchNorm3d(dim))
            encoder_layers.append(nn.ReLU()) #make this adaptive
            input_dim = dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        
        # Decoder
        decoder_layers = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            decoder_layers.append(nn.ConvTranspose3d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=1, padding=1))
            decoder_layers.append(nn.BatchNorm3d(hidden_dims[i+1]))
            decoder_layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_layers)

        self.final_layer = nn.Sequential(nn.ConvTranspose3d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm3d(hidden_dims[-1]),
                                         nn.ReLU(), #maybe change this
                                         nn.Conv2d(hidden_dims[-1], out_channels= 3,kernel_size= 3, padding= 1),
                                         nn.Tanh())

  