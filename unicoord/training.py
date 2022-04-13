import numpy as np
import torch
from .loss_functions import cal_recon_loss,cal_kl_normal_loss,\
    cal_kl_multiple_discrete_loss,\
        cal_diffusion_loss,cal_clustering_loss,cal_reg_loss_sign,\
            cal_cross_entropy,cal_hierarchy_loss
from tqdm import tqdm_notebook

EPS = 1e-12


class Trainer():
    def __init__(self, model, optimizer, 
                 beta=1, 
                 cont_capacity=None,disc_capacity=None,
                 discrete_supervised_lambda = 0, disc_supervise_dict = None, 
                 continuous_supervised_lambda = 0, cont_supervise_dict = None,
                 clustering_lambda=0, cluster_disc_dims = None, ratio_interOverIntra = 0.1, 
                 diffusion_lambda=0, diffuse_cont_dims = None, 
                 sigma = 1, random_walk_step = 1,
                 hierarchy_lambda = 0, cluster_tree = None, tree_layer_to_disc_mapping = None,
                 print_loss_every=100, record_loss_every=5,
                 use_cuda=False, verbose = False, 
                 cont_loss_type = 'order',scaling_loss = False, leave_tqdm = True):
        """
        Class to handle training of model.

        Parameters
        ----------
        model : jointvae.models.VAE instance

        optimizer : torch.optim.Optimizer instance

        cont_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_z).
            Parameters to control the capacity of the continuous latent
            channels. Cannot be None if model.is_continuous is True.
        disc_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_c).
            Parameters to control the capacity of the discrete latent channels.
            Cannot be None if model.is_discrete is True.
        
        discrete_supervised_lambda: float
            Weight of loss for discrete supervision.
            If is 0, no discrete supervision will be applied
        disc_supervise_dict: dict
            Dicionary showing that which latent disc dim will be supervised by which label dim 
            {lat disc dim: label dim}
            If None, all discrete dims will be supervised by the first label
        
        continuous_supervised_lambda: float
            Weight of loss for continuous supervision.
            If is 0, no continuous supervision will be applied
        cont_supervise_dict: dict
            Dicionary showing that which latent cont dim will be supervised by which label dim 
            {lat cont dim: label dim}
            If None, all cont dims will be supervised by the first label
            
        clustering_lambda: float
            Weight of loss for clustering.
            If is 0, no clustering loss will be applied
        cluster_disc_dims: list
            List of disc dims that under constrain of clustering loss.
            If None, all disc dims will be punished
        ratio_interOverIntra: float
            Ratio between weights of inter and intra clustering loss
            
        diffusion_lambda: float
            Weight of loss for diffusion.
            If is 0, no diffusion loss will be applied
        diffuse_cont_dims: list
            List of disc dims that under constrain of clustering loss.
            If None, all disc dims will be punished
        sigma: float
            std of Gaussian kernel, larger sigma yields more neighbor for each sample, 
            used in diffusion loss
        random_walk_step: int
            Step number of random walk by neighboring probablity matrix, 
            used in diffusion loss
            
        print_loss_every : int
            Frequency with which loss is printed during training.
        record_loss_every : int
            Frequency with which loss is recorded during training.
            

        use_cuda : bool
            If True moves model and training to GPU.
        """
        # Parameters
        self.model = model
        self.optimizer = optimizer
        self.beta = beta
        self.cont_capacity = cont_capacity
        self.disc_capacity = disc_capacity
        self.print_loss_every = print_loss_every
        self.record_loss_every = record_loss_every
        self.use_cuda = use_cuda
        self.clustering_lambda = clustering_lambda
        self.cluster_disc_dims = cluster_disc_dims
        self.ratio_interOverIntra = ratio_interOverIntra
        self.diffusion_lambda = diffusion_lambda
        self.diffuse_cont_dims = diffuse_cont_dims
        self.discrete_supervised_lambda = discrete_supervised_lambda
        self.disc_supervise_dict = disc_supervise_dict
        self.continuous_supervised_lambda = continuous_supervised_lambda
        self.cont_supervise_dict = cont_supervise_dict
        self.sigma = sigma
        self.random_walk_step = random_walk_step
        self.verbose = verbose
        self.leave_tqdm = leave_tqdm
        self.cont_loss_type = cont_loss_type
        self.scaling_loss = scaling_loss
        self.hierarchy_lambda = hierarchy_lambda
        self.cluster_tree = cluster_tree
        self.tree_layer_to_disc_mapping = tree_layer_to_disc_mapping
        if not cont_supervise_dict:
            self.unsupervise_cont_dims = [i for i in range(self.model.latent_spec['cont'])]
        else:
            self.unsupervise_cont_dims = [i for i in range(self.model.latent_spec['cont']) if i not in cont_supervise_dict]

        # Initialize attributes
        self.losses = self._check_losses_used()
        self.num_steps = 0
        if self.use_cuda:
            self.model.cuda()

        
        
    def _generate_tree_tables(self,latent_spec, cluster_tree, tree_layer_to_disc_mapping, use_cuda):
        if cluster_tree.depth() <= 1:
            raise RuntimeError("hierarchy loss needs the tree to have at least two layers")
        tree_tables = dict()
        for depth in range(1,cluster_tree.depth()+1):
            layer1_spec = latent_spec['disc'][tree_layer_to_disc_mapping[depth-1]]
            layer2_spec = latent_spec['disc'][tree_layer_to_disc_mapping[depth]]
            layer1_node = [node for node in cluster_tree.all_nodes() if cluster_tree.depth(node=node)==(depth-1)]
            layer2_node = [node for node in cluster_tree.all_nodes() if cluster_tree.depth(node=node)==depth]
            layer1_cluster = [node.data.cell_type_cluster_id for node in layer1_node]
            layer2_clusters = [[node2.data.cell_type_cluster_id for node2 in cluster_tree.children(nid=node1.identifier)] for node1 in layer1_node]
            layer2_clusters_flat = sum(layer2_clusters,[])
            if set(layer1_cluster) != set(range(layer1_spec)) or set(layer2_clusters_flat) != set(range(layer2_spec)):
                raise RuntimeError("number of tree nodes is not the same with number of clusters in latent discrete dim " + str(depth))
            tree_table = np.zeros([len(layer1_cluster), len(layer2_clusters_flat)])
            for idx,cluster1 in enumerate(layer1_cluster):
                for cluster2 in layer2_clusters[idx]:
                    tree_table[cluster1, cluster2] = 1
            tree_table = torch.tensor(tree_table).to(torch.float32).transpose(0,1)
            if use_cuda:
                tree_table = tree_table.cuda()
            tree_tables[depth] = tree_table
        return tree_tables

    def _check_losses_used(self, other_losses = None):
        """
        This function check if the losses used in the trainer is validated,
        and create a dictionary to store all different losses
        """
        if self.model.is_continuous and self.cont_capacity is None:
            raise RuntimeError("Model is continuous but cont_capacity not provided.")
        if self.model.is_discrete and self.disc_capacity is None:
            raise RuntimeError("Model is discrete but disc_capacity not provided.")
        if not self.model.is_continuous:
            if sum([abs(self.continuous_supervised_lambda), abs(self.diffusion_lambda)])>0:
                raise RuntimeError("Model is not continuous but continuous loss is used.")
        if not self.model.is_discrete:
            if sum([abs(self.discrete_supervised_lambda), abs(self.clustering_lambda)])>0:
                raise RuntimeError("Model is not discrete but discrete loss is used.")
        losses = {'loss': [],
               'recon_loss': []}
        # Keep track of divergence values for each latent variable
        if self.model.is_continuous:
            losses['kl_loss_cont'] = []
            # For every dimension of continuous latent variables
            for i in range(self.model.latent_spec['cont']):
                losses['kl_loss_cont_' + str(i)] = []
        if self.model.is_discrete:
            losses['kl_loss_disc'] = []
            # For every discrete latent variable
            for i in range(len(self.model.latent_spec['disc'])):
                losses['kl_loss_disc_' + str(i)] = []
        # loss for unsupervised and other supervised
        if self.discrete_supervised_lambda>0:
            losses['discrete_supervised_loss'] = []
        if self.continuous_supervised_lambda>0:
            losses['continuous_supervised_loss'] = []
        if self.clustering_lambda>0:
            losses['clustering_loss'] = []
        if self.diffusion_lambda>0:
            losses['diffusion_loss'] = []
        # generate transform table from hierarchical tree
        if self.hierarchy_lambda>0:
            if not self.tree_layer_to_disc_mapping or not self.cluster_tree:
                raise RuntimeError("parameters 'tree_layer_to_disc_mapping' and 'cluster_tree' must be provided if hierarchy loss is needed")
            self.tree_tables = self._generate_tree_tables(self.model.latent_spec, self.cluster_tree, 
                                            self.tree_layer_to_disc_mapping, self.use_cuda)
            losses['hierarchy_loss'] = []
        if other_losses:
            for l in other_losses:
                losses[l] = []
        return losses
        
    
    def train(self, data_loader, epochs=10, save_training_gif=None):
        """
        Trains the model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader

        epochs : int
            Number of epochs to train the model for.

        save_training_gif : None or tuple (string, Visualizer instance)
            If not None, will use visualizer object to create image of samples
            after every epoch and will save gif of these at location specified
            by string. Note that string should end with '.gif'.
        """

        self.batch_size = data_loader.batch_size
        self.model.train()
        if self.verbose:
            ran = range(epochs)
        else:
#             if self.using_notebook:
#                 ran = tnrange(epochs,leave = self.leave_tqdm)
#             else:
            ran = tqdm_notebook(range(epochs),leave = self.leave_tqdm)
        for epoch in ran:
            mean_epoch_loss = self._train_epoch(data_loader)
            loss_showed = self.batch_size * self.model.num_pixels * mean_epoch_loss
            if self.verbose:
                print('Epoch: {} Average loss: {:.2f}'.format(epoch + 1,loss_showed))
            else:
                ran.set_description('Epoch ' + str(epoch+1))
                ran.set_postfix(Epoch_average_loss = str('%.2f' % loss_showed))


    def _train_epoch(self, data_loader):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """
        epoch_loss = 0.
        print_every_loss = 0.  # Keeps track of loss to print every
                               # self.print_loss_every
        for batch_idx, (data, label) in enumerate(data_loader):
            iter_loss = self._train_iteration(data, label)
            epoch_loss += iter_loss
            print_every_loss += iter_loss
            # Print loss info every self.print_loss_every iteration
            if batch_idx % self.print_loss_every == 0:
                if batch_idx == 0:
                    mean_loss = print_every_loss
                else:
                    mean_loss = print_every_loss / self.print_loss_every
                if self.verbose:
                    print('{}/{}\tLoss: {:.3f}'.format(batch_idx * len(data),
                                           len(data_loader.dataset),
                                           self.model.num_pixels * mean_loss))
                print_every_loss = 0.
        # Return mean epoch loss
        return epoch_loss / len(data_loader.dataset)

    def _train_iteration(self, data,label):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            A batch of data. Shape (N, C, G)
        """
        self.num_steps += 1
        # if self.use_cuda:
        #     data = data.cuda()

        self.optimizer.zero_grad()
        recon_batch, latent_dist = self.model(data.view(data.size()[0],-1))
        loss = self._loss_function(data, recon_batch, latent_dist,label)
        loss.backward()
        self.optimizer.step()

        train_loss = loss.item()
        return train_loss

    def _loss_function(self, data, recon_data, latent_dist,label):
        """
        Calculates loss for a batch of data.
        Loss comes from several sources, a conventional VAE model's loss have 2 parts:
            1. reconstruction loss: difference between reconstructed data and original data
            2. KL divergences: difference between latent distribution and a prior
        In joint VAE, more kinds of losses were intruduced, including:
            1. reconstruction loss
            2. KL divergences:
                2.1 KL of discret latent dims
                2.2 KL of continuous latent dims
            3. clustering loss: unsupervised, for disc dims, to 
                3.1 reward variance between clusters and 
                3.2 punish variance within cluster
            4. diffusion loss: unsupervised, for cont dims, encourage a graduate change in certain diffusion dims
            5. supervised loss: difference between certain latent dims to labels they are designed to learn

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Should have shape (N, C, G)

        recon_data : torch.Tensor
            Reconstructed data. Should have shape (N, C, G)

        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both containing the parameters
            of the latent distributions as values.
        """
        loss_dict = dict()
        
        # Reconstruction loss is pixel wise cross-entropy
        loss_dict['recon_loss'] = cal_recon_loss(data, recon_data, self.model.num_pixels)
        # KL divergences
        kl_loss_dict = dict()
        if self.model.is_continuous:
            mean, logvar = latent_dist['cont']
            cont_kl_results = cal_kl_normal_loss(mean, logvar, self.cont_capacity, 
                                     self.num_steps)
            cont_capacity_loss, kl_loss_cont, kl_losses_cont = cont_kl_results
            loss_dict['cont_capacity_loss'] = self.beta*cont_capacity_loss
            # save KL divergences for all and each cont dim
            kl_loss_dict['kl_loss_cont'] = kl_loss_cont
            for i in range(self.model.latent_spec['cont']):
                kl_loss_dict['kl_loss_cont_' + str(i)] = kl_losses_cont[i]
        if self.model.is_discrete:
            disc_kl_results = cal_kl_multiple_discrete_loss(latent_dist['disc'], self.disc_capacity, 
                                            self.model.latent_spec['disc'], self.num_steps, self.use_cuda)
            disc_capacity_loss, kl_loss_disc, kl_losses_disc = disc_kl_results
            loss_dict['disc_capacity_loss'] = self.beta*disc_capacity_loss
            # save KL divergences for all and each disc dim
            kl_loss_dict['kl_loss_disc'] = kl_loss_disc
            for i in range(len(self.model.latent_spec['disc'])):
                kl_loss_dict['kl_loss_disc_' + str(i)] = kl_losses_disc[i]
        # diffusion/clustering loss
        if self.diffusion_lambda >0:
            diffusion_loss = cal_diffusion_loss(latent_dist, self.diffuse_cont_dims, self.unsupervise_cont_dims, 
                                    data, self.model, self.use_cuda, self.sigma, self.random_walk_step)
            diffusion_loss *= self.diffusion_lambda
            loss_dict['diffusion_loss'] = diffusion_loss
        if self.clustering_lambda > 0:
            clustering_loss = cal_clustering_loss(latent_dist['disc'], self.cluster_disc_dims, 
                                      data, self.ratio_interOverIntra)
            clustering_loss *= self.clustering_lambda
            loss_dict['clustering_loss'] = clustering_loss
        # Supervised loss
        if self.continuous_supervised_lambda > 0:
            continuous_supervised_loss = cal_reg_loss_sign(latent_dist['cont'], label, 
                                            self.cont_supervise_dict, self.use_cuda,
                                            loss_type = self.cont_loss_type, scaling_loss = self.scaling_loss)
            continuous_supervised_loss *= self.continuous_supervised_lambda
            loss_dict['continuous_supervised_loss'] = continuous_supervised_loss
        if self.discrete_supervised_lambda > 0:
            discrete_supervised_loss = cal_cross_entropy(latent_dist['disc'], label,
                                            self.disc_supervise_dict, self.use_cuda)
            discrete_supervised_loss *= self.discrete_supervised_lambda
            loss_dict['discrete_supervised_loss'] = discrete_supervised_loss
        if self.hierarchy_lambda > 0:
            hierarchy_loss = cal_hierarchy_loss(self.tree_tables, latent_dist['disc'], 
                                    self.tree_layer_to_disc_mapping)
            hierarchy_loss *= self.hierarchy_lambda
            loss_dict['hierarchy_loss'] = hierarchy_loss
        # Total loss
        total_loss = 0    
        for loss_name in loss_dict:
            total_loss += loss_dict[loss_name]
        loss_dict['loss'] = total_loss
        
        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            for loss_name in self.losses:
                if loss_name.startswith('kl_loss_'):
                    self.losses[loss_name].append(kl_loss_dict[loss_name].item())
                else:
                    self.losses[loss_name].append(loss_dict[loss_name].item())

        # To avoid large losses normalise by number of pixels
        return total_loss / self.model.num_pixels
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        