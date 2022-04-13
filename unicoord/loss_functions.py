from .utils import sample_gumbel_softmax, pairwise_distances, EPS


import torch
from torch.nn import functional as F
import numpy as np

def cal_recon_loss(data, recon_data, num_pixels):
    recon_loss = F.mse_loss(recon_data.view(-1, num_pixels),
                    data.view(-1, num_pixels))
    # F.binary_cross_entropy takes mean over pixels, so unnormalise this
    recon_loss *= num_pixels
    return recon_loss

def cal_kl_normal_loss(mean, logvar, cont_capacity, num_steps):
    """
    Calculates the KL divergence between a normal distribution with
    diagonal covariance and a unit normal distribution.
    
    Capacity is to linearly increse the weight of KL loss

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (N, D) where D is dimension
        of distribution.
    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (N, D)
    cont_capacity: tuple (float, float, int, float)
    """
    # Calculate KL divergence
    kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    # Mean KL divergence across batch for each latent variable
    kl_means = torch.mean(kl_values, dim=0)
    # KL loss is sum of mean KL of each latent variable
    kl_loss = torch.sum(kl_means)
    # Increase continuous capacity without exceeding cont_max
    cont_min, cont_max, cont_num_iters, cont_gamma = cont_capacity
    cont_cap_current = (cont_max - cont_min) * num_steps / float(cont_num_iters) + cont_min
    cont_cap_current = min(cont_cap_current, cont_max)
    # Calculate continuous capacity loss
    cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_loss)
    return cont_capacity_loss, kl_loss, kl_means

def cal_kl_multiple_discrete_loss(alphas, disc_capacity, latent_spec_disc, num_steps, use_cuda):
    """
    Calculates the KL divergence between a set of categorical distributions
    and a set of uniform categorical distributions.

    Parameters
    ----------
    alphas : list
        List of the alpha parameters of a categorical (or gumbel-softmax)
        distribution. For example, if the categorical atent distribution of
        the model has dimensions [2, 5, 10] then alphas will contain 3
        torch.Tensor instances with the parameters for each of
        the distributions. Each of these will have shape (N, D).
    disc_capacity: tuple (float, float, int, float)
    latent_spec_disc: list[int]
    """
    # Calculate kl losses for each discrete latent
    kl_losses = [cal_kl_discrete_loss(alpha, use_cuda) for alpha in alphas]
    # Total loss is sum of kl loss for each discrete latent
    kl_loss = torch.sum(torch.cat(kl_losses))
    disc_min, disc_max, disc_num_iters, disc_gamma = disc_capacity
    # Increase discrete capacity without exceeding disc_max or theoretical
    disc_cap_current = (disc_max - disc_min) * num_steps / float(disc_num_iters) + disc_min
    disc_cap_current = min(disc_cap_current, disc_max)
    # Require float conversion here to not end up with numpy float
    disc_theoretical_max = sum([float(np.log(disc_dim)) for disc_dim in latent_spec_disc])
    disc_cap_current = min(disc_cap_current, disc_theoretical_max)
    # Calculate discrete capacity loss
    disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_loss)
    return disc_capacity_loss, kl_loss, kl_losses


log_dim_dict = dict()
def cal_kl_discrete_loss(alpha, use_cuda):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.

    Parameters
    ----------
    alpha : torch.Tensor
        Parameters of the categorical or gumbel-softmax distribution.
        Shape (N, D)
    """
    disc_dim = int(alpha.size()[-1])
    if disc_dim not in log_dim_dict:
        log_dim = torch.Tensor([np.log(disc_dim)])
        if use_cuda:
            log_dim = log_dim.cuda()
        log_dim_dict[disc_dim] = log_dim
    log_dim = log_dim_dict[disc_dim]
    # Calculate negative entropy of each row
    neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
    # Take mean of negative entropy across batch
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    # KL loss of alpha with uniform categorical variable
    kl_loss = log_dim + mean_neg_entropy
    return kl_loss

def cal_diffusion_loss(latent_dist, diffuse_cont_dims, unsupervise_cont_dims,
                data, model, use_cuda, sigma=1, random_walk_step = 1):
    """
    Calculated loss for trajectory
    Assuming that a continuous latent distribution stores the trajectory information, which means similar value in 
    this latent dimension implys similar original data. 

    Using the difference between 
        1.original data of a sample, 
        2.data reconstructed from its neighbor's latent distribution, 
    as loss

    Parameters
    ----------
    latent_dist : dict
        Dict with keys 'cont' or 'disc' or both containing the parameters
        of the latent distributions as values.
    diffuse_cont_dims: list[int]
        Indexes of latent cont dims that constrained by diffusion loss
    data: torch.Tensor
        Original data. Shape (N,G)
    model: torch.nn
        Neural network of VAE model
    sigma: float
        std of Gaussian kernel, larger sigma yields more neighbor for each sample 
    random_walk_step: int
        Step number of random walk by neighboring probablity matrix
    """
    # if diffuse_cont_dims is not set, use all continuous dims
    if not diffuse_cont_dims:
        diffuse_cont_dims = range(0,latent_dist['cont'][0].shape[1])
    # mean, data, recon_data should all be in cuda mode
    mean = latent_dist['cont'][0][:,diffuse_cont_dims]
    mean = mean.cpu()
    # calculate distance matrix in diffusion space for every pair of data
    z_dist = pairwise_distances(mean)
    # use a gaussian kernel and sofemax normalization, 
    # let remote data have lower probablity to be neighbors
    # neg_z_dist = torch.exp(-torch.mul(z_dist,z_dist)/sigma)
    neg_z_dist = torch.exp(-z_dist/sigma)
    exclude_self_mat = torch.ones(len(mean),len(mean)) - torch.eye(len(mean))
    neg_z_dist = torch.mul(neg_z_dist, exclude_self_mat)
    if use_cuda:
        neg_z_dist = neg_z_dist.cuda()
    neighbor_prob = sample_gumbel_softmax(neg_z_dist, model.temperature, 
                              model.training, use_cuda)
    # a random walk will be applied if required
    if random_walk_step > 1:
        one_step_prob = neighbor_prob.detach()
        for i in range(random_walk_step):
            neighbor_prob = torch.mm(one_step_prob, neighbor_prob)
    # diffusion is applied on all continuous dims, 
    # but only diffuse_cont_dims will have loss back propagation from diffusion loss 
#     diffused_ld = latent_dist
#     diffused_ld['cont'][0] = torch.mm(neighbor_prob, 
#                            latent_dist['cont'][0])
#     diffused_ld['cont'][1] = torch.mm(neighbor_prob, 
#                            latent_dist['cont'][1])    
    diffused_ld = {'cont': [torch.tensor(np.array(x.detach().cpu())) for x in latent_dist['cont']],
              'disc': [torch.tensor(np.array(x.detach().cpu())) for x in latent_dist['disc']]}
    if use_cuda:
        diffused_ld = {'cont': [x.cuda() for x in diffused_ld['cont']],
                  'disc': [x.cuda() for x in diffused_ld['disc']]}
    diffused_ld['cont'][0][:,unsupervise_cont_dims] = torch.mm(neighbor_prob, 
                                            latent_dist['cont'][0][:,unsupervise_cont_dims])
    diffused_ld['cont'][1][:,unsupervise_cont_dims] = torch.mm(neighbor_prob, 
                                            latent_dist['cont'][1][:,unsupervise_cont_dims])
    # after diffusion, use diffused latent distribution to do sampling and decoding,
    # calculate reconstruction error
    neighbor_recon_data = model.decode(model.reparameterize(diffused_ld))
    recon_loss = F.mse_loss(neighbor_recon_data.view(-1, model.num_pixels),
                 data.view(-1, model.num_pixels))
    recon_loss *= model.num_pixels
    return recon_loss 

def cal_clustering_loss(latent_dist_disc, cluster_disc_dims, data, ratio_interOverIntra):
    """
    Calculates the self-superviseed loss in clustering term
    
    Parameters:
    -----------
    latent_dist_disc: list[torch.tensor]
        Probability weight matrixs for latent disc distributions
    cluster_disc_dims: list[int]
        Indexes of latent disc dims that constrained by clustering loss
    data: torch.Tensor
        Original data. Shape (N,G)
    ratio_interOverIntra: float
        Ratio between weights of inter and intra clustering loss,
        If 1, equal weight will be applied, 
        higher ratio yeilds heigher weight of inter cluster loss
        
    """
    # if cluster_disc_dims is not set, use all discrete dims
    if not cluster_disc_dims:
        cluster_disc_dims = range(0,len(latent_dist_disc))
    clustering_loss = 0
    dist = pairwise_distances(data)
    dist = dist ** 2
    for d in cluster_disc_dims:
        alpha = latent_dist_disc[d]
        q = torch.mm(alpha,alpha.transpose(0,1))
        # transfer (0,1) to (-ratio,1)
        # so that intra cluster distances are punished by weight 1, 
        # inter cluster distances are rewarded by weight ratio
        clustering_loss += torch.mean(dist*(q*(ratio_interOverIntra + 1)-ratio_interOverIntra))
    return clustering_loss

def cal_reg_loss_sign(latent_dist_cont, label, cont_supervise_dict, 
               use_cuda, loss_type, scaling_loss, factor=1.0):
    """
    Computes the regularization loss given the latent code and attribute
    
    Parameters
    ----------
    latent_dist_cont: list[torch.tensor]
        Latent distribution parameters of continuous dims, shape should be (2, N, D)
    label: torch.tensor
        labels used as supervision shape should be (N, L) where L is the number of labels
    cont_supervise_dict:dict
        Dicionary showing that which latent cont dim will be supervised by which label dim 
        {lat cont dim: label dim}
        If None, all cont dims will be supervised by the first label
    loss_type: str
        either "order" or 'mse'
    scaling_loss: boolean
        Wether to adjust loss of each supervised dimention according to data scale
    factor: float
        Factor multiplied onto latent distance matrix
    """
    # if cont_supervise_dict is not set, all cont dims will be supervised by the first label
    if not cont_supervise_dict:
        cont_supervise_dict = {i:0 for i in range(0,latent_dist_cont[0].shape[1])}
    mean, logvar = latent_dist_cont
    
    reg_losses = []
    attri_scale = []
    for latent in cont_supervise_dict:
        # the lth (not one, L)latent dim is supervised by the cont_supervise_dict[l]th label
        alpha = mean[:,latent]
        attribute = label[:,cont_supervise_dict[latent]]
        attribute = attribute.to(torch.float32)
        attri_scale.append(torch.mean(attribute.detach().cpu()))
        if loss_type == 'order':
            # compute latent distance matrix
            alpha = alpha.view(-1, 1).repeat(1, alpha.shape[0])
            lc_dist_mat = (alpha - alpha.transpose(1, 0)).view(-1, 1)

            # compute attribute distance matrix
            attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
            attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

            # compute regularization loss
            loss_fn = torch.nn.L1Loss()
            lc_tanh = torch.tanh(lc_dist_mat * factor)
            attribute_sign = torch.sign(attribute_dist_mat)
            if use_cuda:
                attribute_sign = attribute_sign.cuda()
            dim_loss = loss_fn(lc_tanh, attribute_sign.float())
        else:
            if use_cuda:
                attribute = attribute.cuda()
            dim_loss = F.mse_loss(alpha, 
                           attribute)
        reg_losses.append(dim_loss)
    reg_loss = 0
    attri_scale = np.array(attri_scale) + 0.1
    for i in range(len(reg_losses)):
        if scaling_loss:
            reg_loss += reg_losses[i] * 1/attri_scale[i]
        else:
            reg_loss += reg_losses[i]
    return reg_loss

def cal_cross_entropy(latent_dist_disc, label,disc_supervise_dict, use_cuda):
    """ 
    Computes the regularization loss given the latent disc dims and labels
    
    Parameters:
    -----------
    latent_dist_disc: list[torch.tensor]
        Probability weight matrixs for latent disc distributions
    label: torch.tensor
        labels used as supervision shape should be (N, L) where L is the number of labels
    disc_supervise_dict: dict
        Dicionary showing that which latent disc dim will be supervised by which label dim 
        {lat disc dim: label dim}
        If None, all discrete dims will be supervised by the first label
    """
    # if disc_supervise_dict is not set, all disc dims will be supervised by the first label
    if not disc_supervise_dict:
        disc_supervise_dict = {i:0 for i in range(0,len(latent_dist_disc))}
    cross_entropy = 0
    for latent in disc_supervise_dict:
        #print(latent, disc_supervise_dict[latent])
        alpha = latent_dist_disc[latent]
        attribute = label[:,disc_supervise_dict[latent]]
        attribute = attribute.type(torch.int64)
        if use_cuda:
            attribute = attribute.cuda()
        ce = F.cross_entropy(alpha,attribute)
        cross_entropy += ce
    return cross_entropy


def cal_hierarchy_loss(tree_tables, latent_dist_disc, tree_layer_to_disc_mapping):
    '''
    Loss constraining the tree structure the discrete layers 
    
    Parameters
    ----------
    cluster_tree: dict{NodeInLayerN:NodesInLayerN+1}
        dictionary that discribing the inclusion relation between adjacent layers
        a key of the dict should be a node in one layer,
        its value should be nodes in the last layer that belongs to the key node
    latent_dist_disc: 
        latent distribution of discrete dimensions
    '''
    hierarchy_loss = 0
    for depth in tree_tables:
        layer1_dist = latent_dist_disc[tree_layer_to_disc_mapping[depth-1]]
        layer2_dist = latent_dist_disc[tree_layer_to_disc_mapping[depth]]
        empirical_layer1 = torch.mm(layer2_dist,tree_tables[depth])
        layer_loss = F.mse_loss(layer1_dist, empirical_layer1)
        hierarchy_loss += layer_loss
    return hierarchy_loss





