"""Official MMIN network classes, MIT licensed; see upstream/LICENSE.
Source: AIM3-RUC/MMIN master 69a45fedff66fa0da74205ad425440c1ee8f6d1f.
Classes copied without algorithmic modification.
"""

import copy

import torch

import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class FcClassifier(nn.Module):
    def __init__(self, input_dim, layers, output_dim, dropout=0.3, use_bn=False):
        ''' Fully Connect classifier
            Parameters:
            --------------------------
            input_dim: input feature dim
            layers: [x1, x2, x3] will create 3 layers with x1, x2, x3 hidden nodes respectively.
            output_dim: output feature dim
            activation: activation function
            dropout: dropout rate
        '''
        super().__init__()
        self.all_layers = []
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            if use_bn:
                self.all_layers.append(nn.BatchNorm1d(layers[i]))
            if dropout > 0:
                self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        
        if len(layers) == 0:
            layers.append(input_dim)
            self.all_layers.append(Identity())
        
        self.fc_out = nn.Linear(layers[-1], output_dim)
        self.module = nn.Sequential(*self.all_layers)
    
    def forward(self, x):
        feat = self.module(x)
        out = self.fc_out(feat)
        return out, feat

class ResidualAE(nn.Module):
    ''' Residual autoencoder using fc layers
        layers should be something like [128, 64, 32]
        eg:[128,64,32]-> add: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (64, 128), (128, input_dim)]
                          concat: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (128, 128), (256, input_dim)]
    '''
    def __init__(self, layers, n_blocks, input_dim, dropout=0.5, use_bn=False):
        super(ResidualAE, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.transition = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        for i in range(n_blocks):
            setattr(self, 'encoder_' + str(i), self.get_encoder(layers))
            setattr(self, 'decoder_' + str(i), self.get_decoder(layers))
    
    def get_encoder(self, layers):
        all_layers = []
        input_dim = self.input_dim
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.LeakyReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
            input_dim = layers[i]
        # delete the activation layer of the last layer
        decline_num = 1 + int(self.use_bn) + int(self.dropout > 0)
        all_layers = all_layers[:-decline_num]
        return nn.Sequential(*all_layers)
    
    def get_decoder(self, layers):
        all_layers = []
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()
        decoder_layer.append(self.input_dim)
        for i in range(0, len(decoder_layer)-2):
            all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i+1]))
            all_layers.append(nn.ReLU()) # LeakyReLU
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
        
        all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1]))
        return nn.Sequential(*all_layers)

    
    def forward(self, x):
        x_in = x
        x_out = x.clone().fill_(0)
        latents = []
        for i in range(self.n_blocks):
            encoder = getattr(self, 'encoder_' + str(i))
            decoder = getattr(self, 'decoder_' + str(i))
            x_in = x_in + x_out
            latent = encoder(x_in)
            x_out = decoder(latent)
            latents.append(latent)
        latents = torch.cat(latents, dim=-1)
        return self.transition(x_in+x_out), latents
