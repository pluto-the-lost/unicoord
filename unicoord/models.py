import torch
from torch import nn
from torch.nn import functional as F
from .utils import sample_normal,sample_gumbel_softmax

EPS = 1e-12

class VAE(nn.Module):
    def __init__(self, data_size, latent_spec, temperature=.67, use_cuda=False, use_cnn = False):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        data_size : 1 dim tuple of ints, dim number of each cell
            Size of data. E.g. (8000, 2000) or (10000, 30000).

        latent_spec : dict
            Specifies latent distribution. For example:
            {'cont': 10, 'disc': [10, 4, 3]} encodes 10 normal variables and
            3 gumbel softmax variables of dimension 10, 4 and 3. A latent spec
            can include both 'cont' and 'disc' or only 'cont' or only 'disc'.

        temperature : float
            Temperature for gumbel softmax distribution.

        use_cuda : bool
            If True moves model to GPU
        """
        super(VAE, self).__init__()
        self.use_cuda = use_cuda

        # Parameters
        self.data_size = data_size
        self.is_continuous = 'cont' in latent_spec
        self.is_discrete = 'disc' in latent_spec
        self.latent_spec = latent_spec
        self.num_pixels = data_size[0] * data_size[1]
        self.temperature = temperature
        self.hidden_dim = 128  # Hidden dimension of linear layer
        self.use_cnn = use_cnn

        # Calculate dimensions of latent distribution
        self.latent_cont_dim = 0
        self.latent_disc_dim = 0
        self.num_disc_latents = 0
        if self.is_continuous:
            self.latent_cont_dim = self.latent_spec['cont']
        if self.is_discrete:
            self.latent_disc_dim += sum([dim for dim in self.latent_spec['disc']])
            self.num_disc_latents = len(self.latent_spec['disc'])
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

        # Define encoder layers that from input to hidden layer 
        # which will be used to infer latent parameters
        
        
        
        if self.use_cnn:
            encoder_layers = [
                un_Flatten(inter_dim = 28),
                nn.Conv2d(1, 64, 4, 1),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.Conv2d(64, 64, 4, 1),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.Conv2d(64, 8, 4, 1),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.Flatten(),
                nn.Linear(2888, self.hidden_dim),
                nn.SELU()
            ]
            decoder_layers = [
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.SELU(),
                nn.Linear(self.hidden_dim, 2888),
                nn.SELU(),
                un_Flatten(inter_dim = 19),
                nn.ConvTranspose2d(8, 64, 4, 1),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.ConvTranspose2d(64, 64, 4, 1),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.ConvTranspose2d(64, 1, 4, 1),
                nn.Flatten()
            ]
        else:
            encoder_layers = [
                nn.Linear(self.num_pixels,512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, self.hidden_dim),
                nn.ReLU()
            ]
            decoder_layers = [
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.num_pixels),
                nn.ReLU()
            ]
        
        self.data_to_hidden = nn.Sequential(*encoder_layers)

        # Encode parameters of latent distribution
        if self.is_continuous:
            self.fc_mean = nn.Linear(self.hidden_dim, self.latent_cont_dim)
            self.fc_log_var = nn.Linear(self.hidden_dim, self.latent_cont_dim)
        if self.is_discrete:
            # Linear layer for each of the categorical distributions
            fc_alphas = []
            for disc_dim in self.latent_spec['disc']:
                fc_alphas.append(nn.Linear(self.hidden_dim, disc_dim))
            self.fc_alphas = nn.ModuleList(fc_alphas)

        # Define decoder
        self.latent_to_recon_data = nn.Sequential(*decoder_layers)

        if use_cuda:
            self.to('cuda')
        
    def encode(self, x):
        """
        Encodes a cell vector into parameters of a latent distribution defined in
        self.latent_spec.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data, shape (N, C, G). N: batch size, C: cell number, G: gene number
        """
        batch_size = x.size()[0]

        # Encode image to hidden features
        hidden = self.data_to_hidden(x)

        # Output parameters of latent distribution from hidden representation
        latent_dist = {}

        if self.is_continuous:
            latent_dist['cont'] = [self.fc_mean(hidden), self.fc_log_var(hidden)]
        if self.is_discrete:
            latent_dist['disc'] = []
            for fc_alpha in self.fc_alphas:
                latent_dist['disc'].append(F.softmax(fc_alpha(hidden), dim=1))

        return latent_dist
    
    def reparameterize(self, latent_dist):
        """
        Samples from latent distribution using the reparameterization trick.

        Parameters
        ----------
        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both, containing the parameters
            of the latent distributions as torch.Tensor instances.
        """
        latent_sample = []

        if self.is_continuous:
            mean, logvar = latent_dist['cont']
            cont_sample = sample_normal(mean, logvar,self.training, self.use_cuda)
            latent_sample.append(cont_sample)

        if self.is_discrete:
            for alpha in latent_dist['disc']:
                disc_sample = sample_gumbel_softmax(alpha,self.temperature, self.training, self.use_cuda)
                latent_sample.append(disc_sample)

        # Concatenate continuous and discrete samples into one large sample
        return torch.cat(latent_sample, dim=1)

    
    
    def decode(self, latent_sample):
        """
        Decodes sample from latent distribution into an image.

        Parameters
        ----------
        latent_sample : torch.Tensor
            Sample from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        return self.latent_to_recon_data(latent_sample)
    
    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (N, C, G)
        """
        latent_dist = self.encode(x)
        latent_sample = self.reparameterize(latent_dist)
        return self.decode(latent_sample), latent_dist
    
class un_Flatten(nn.Module):
    def __init__(self, inter_dim):
        super(un_Flatten,self).__init__()
        self.inter_dim = inter_dim
 
    def forward(self,input):
        return input.view(input.size(0), -1, self.inter_dim, self.inter_dim)