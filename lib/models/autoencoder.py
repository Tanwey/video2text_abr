import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, latent):
        super(Autoencoder, self).__init__()
        self.latent = latent
        self.encoder_linear1 = nn.Linear(1024, latent)
        self.decoder_linear1 = nn.Linear(latent, 1024)
        
    def encode(self, x):
        x = F.leaky_relu(self.encoder_linear1(x))
        return x
    
    def decode(self, x):
        x = self.encoder_linear1(x)
        return x
        
    def __call__(self, x):
        x = self.encoder_linear1(x)
        x = self.decoder_linear1(x)
        return x