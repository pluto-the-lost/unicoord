import numpy as np
import pandas as pd
import scanpy as sc
from .models import VAE
from .training import Trainer
import random

from torch import optim
import torch
from torch.utils.data import TensorDataset,DataLoader
from itertools import chain
from collections import OrderedDict

gpu = torch.cuda.is_available()

def model_unicoord_in_adata(adata, obs_fitting = None, min_obs = 200,
                            genes_used = None, use_highly_variable = False,
                              n_disc = None, n_clus = None, n_cont = 20, n_diff = 0):
    obs_fitting = [] if obs_fitting is None else obs_fitting
    n_disc = [] if n_disc is None else n_disc
    n_clus = [] if n_clus is None else n_clus
 
    unc_stuffs = dict()
    if not genes_used:
        if use_highly_variable:
            genes_used = list(adata.var_names[adata.var.highly_variable])
        else:
            genes_used = list(adata.var_names)
    unc_stuffs['genes_used'] = genes_used
 


    obs_type = {k:pd.api.types.is_numeric_dtype(adata.obs[k]) for k in obs_fitting}
    def _disc_mapping(l):
        counts = l.value_counts()
        rare_obs = set(counts[counts < min_obs].keys())
        disc_mapping = dict(zip(counts.keys(), range(1,1+len(counts))))
        for c in disc_mapping: 
            if c in rare_obs: 
                disc_mapping[c]=0 
        return disc_mapping

    obs_type = {'cont': [k for k,v in obs_type.items() if v],
                'disc': {k:max(_disc_mapping(adata.obs[k]).values())+1 \
                                for k,v in obs_type.items() if not v}}
    unc_stuffs['obs_fitting'] = obs_type

    unc_stuffs['disc_mapping'] = {k:_disc_mapping(adata.obs[k]) for k in obs_type['disc']}
    # unc_stuffs['disc_mapping'] = {k:{key:idx for idx,key in \
    #                                 enumerate(adata.obs[k].value_counts().index)\
    #                                 if adata.obs[k].value_counts()[key]>200 else -1} \
    #                               for k in obs_type['disc']}

    if n_clus and isinstance(n_clus, int):
        n_clus = [n_clus]

    unc_stuffs['n_dims'] = {'cont':{'n_cont' : n_cont, 'n_diff' : n_diff,
                                    'n_cont_sup' : len(obs_type['cont'])},
                            'disc':{'n_disc' : n_disc, 'n_clus' : n_clus, 
                                    'n_disc_sup' : list(obs_type['disc'].values())}}

    unc_stuffs['dim_definition'] = {'cont': ['sup']*len(obs_type['cont']) + \
                                            ['diff']*n_diff + ['unsup']*n_cont,
                                    'disc': ['sup']*len(obs_type['disc']) + \
                                            ['clus']*len(n_clus) + ['unsup']*len(n_disc)}


    latent_spec = {'cont': sum(unc_stuffs['n_dims']['cont'].values()), 
                   'disc': unc_stuffs['n_dims']['disc']['n_disc_sup'] + n_clus + n_disc}
    if not latent_spec['cont']: del latent_spec['cont']
    if not latent_spec['disc']: del latent_spec['disc']


    model = VAE(latent_spec=latent_spec, 
                data_size = (len(genes_used),1), 
                use_cuda = gpu)
    model.training = False
    unc_stuffs['model'] = model
    adata.uns['unc_stuffs'] = unc_stuffs

def _tensor_from_adata(adata, cells_chunk, genes_used, slot, label_needed = True, unc_stuffs = None):
    # print(len(adata.var_names), len(genes_used))
    absent_genes = set(genes_used) - set(adata.var_names)
    if absent_genes:
        print('%d needed genes are not exist in the query adata, filled with zeros'%(len(absent_genes)))
        tmp = sc.AnnData(X=np.zeros((1, len(genes_used))), var=pd.DataFrame(index=genes_used))
        adata = adata.concatenate(tmp, join = 'outer', index_unique = None)[:len(adata.obs_names), :]
        adata.var['absent_genes'] = False
        adata.var.absent_genes.loc[absent_genes] = True

    for cells in cells_chunk:
        if slot == 'raw':
            mtx = adata.raw[cells,genes_used].X
        elif slot == 'cur':
            mtx = adata[cells,genes_used].X
        try:
            mtx = mtx.toarray()
        except AttributeError:
            pass
        mtx = torch.FloatTensor(mtx)
        if gpu:
            mtx = mtx.cuda()
        if label_needed:
            if not unc_stuffs:
                unc_stuffs = adata.uns['unc_stuffs']
            # training label
            obs_fitting = unc_stuffs['obs_fitting']
            disc_mapping = unc_stuffs['disc_mapping']
            label = adata[cells,:].obs.loc[:,obs_fitting['cont']]
            for k in obs_fitting['disc']:
                label[k] = [disc_mapping[k][i] for i in adata[cells,:].obs[k]]
            label = torch.tensor(np.array(label))
            if gpu:
                label = label.cuda()

            dataset_torch = TensorDataset(mtx,label)
        else:
            dataset_torch = TensorDataset(mtx)

        yield dataset_torch

def train_unicoord_in_adata(adata, unc_stuffs = None,
                            train_with_all = False,
                            epochs = 100, chunk_size = 20000,
                            loss_weights = None, 
                              optimizer = None,
                              slot = 'cur', keep_bar = True):
    if not unc_stuffs:
        unc_stuffs = adata.uns['unc_stuffs']
        
    model = unc_stuffs['model']
    model.training = True

    if not loss_weights:
        if 'loss_weights' not in unc_stuffs:
            loss_weights = {'clus':0.001, 'diff':0.001, 'disc':100, 'cont':100, 
                            'cont_capacity':[0.0, 5.0, 25000, 30.0], 
                            'disc_capacity':[0.0, 5.0, 25000, 30.0]}
        else:
            loss_weights = unc_stuffs['loss_weights']
    unc_stuffs['loss_weights'] = loss_weights

    if not optimizer:
        if 'optimizer' not in unc_stuffs:
            optimizer = optim.Adam(model.parameters(), lr=5e-4)
        else:
            optimizer = unc_stuffs['optimizer']

    if 'unc_training' not in adata.obs_keys():
        n_training = round(len(adata.obs_names) * 0.8)
        training_idx = [True]*n_training + [False]*(len(adata.obs_names) - n_training)
        random.shuffle(training_idx)
        adata.obs.loc[:,'unc_training'] = training_idx

    unc_stuffs['slot'] = slot

    # training matrix
    if train_with_all:
        cells_used = adata.obs_names
    else:
        cells_used = adata.obs_names[adata.obs.unc_training]
    genes_used = unc_stuffs['genes_used']
    
    # training processes
    clus,diff,disc,cont,cont_capacity,disc_capacity = loss_weights.values()
    diffuse_cont_dims = [i for i,dim in enumerate(unc_stuffs['dim_definition']['cont']) if dim =='diff']
    cont_supervise_dict = {i:i for i,dim in enumerate(unc_stuffs['dim_definition']['cont']) if dim =='sup'}
    cluster_disc_dims = [i for i,dim in enumerate(unc_stuffs['dim_definition']['disc']) if dim =='clus']
    # first dims of label are continuous, followed by discret
    disc_supervise_dict = {i:i+len(cont_supervise_dict) for i,dim in \
                            enumerate(unc_stuffs['dim_definition']['disc']) if dim =='sup'}
    if len(cont_supervise_dict) == 0:
        cont = 0
    if len(disc_supervise_dict) == 0:
        disc = 0
    if len(cluster_disc_dims) == 0:
        clus = 0
    if len(diffuse_cont_dims) == 0:
        diff = 0

    trainer = Trainer(model, optimizer,
                      cont_capacity=cont_capacity,disc_capacity=disc_capacity,
                      clustering_lambda = clus, cluster_disc_dims = cluster_disc_dims,
                      diffusion_lambda=diff, diffuse_cont_dims = diffuse_cont_dims, 
                      sigma = 5, random_walk_step = 1,
                      discrete_supervised_lambda = disc, disc_supervise_dict = disc_supervise_dict,
                      continuous_supervised_lambda = cont, cont_supervise_dict = cont_supervise_dict,
                      cont_loss_type = 'mse', scaling_loss = False, 
                      print_loss_every=100, record_loss_every=5,
                      use_cuda = gpu, verbose = False, leave_tqdm = keep_bar)

    cells_used = list(adata.obs_names)
    random.shuffle(cells_used)
    cells_chunk = [list(cells_used[i:i+chunk_size]) \
                   for i in range(0,len(cells_used),chunk_size)]

    train = _tensor_from_adata(adata, cells_chunk, genes_used, slot, unc_stuffs=unc_stuffs)
    for idx, data in enumerate(train):
        print("training chunk %d / %d of the data"%(idx+1, len(cells_chunk)))
        train_loader = DataLoader(data, batch_size=128,shuffle=True)
        trainer.train(train_loader, epochs = epochs)
    model.training = False

    # save loss curves
    cur_loss = trainer.losses
    if "loss" not in unc_stuffs:
        unc_stuffs["loss"] = dict()
    for l in cur_loss:
        if l not in unc_stuffs["loss"]:
            unc_stuffs["loss"][l] = []
        unc_stuffs["loss"][l].extend(cur_loss[l])
    for l in unc_stuffs["loss"]:
        if l not in cur_loss:
            unc_stuffs["loss"][l].extend([0]*len(cur_loss['loss']))
    

def _find_posterior(data, model):
    '''
    find posterior for each batch in a dataset and concat all of them afterward
    '''
    for data in DataLoader(data, batch_size=len(data),shuffle=False):
        x = data[0].view(data[0].size(0),-1)
        with torch.no_grad():
            recon_sample,latent_dist = model(x)

    return recon_sample,latent_dist

    # if model.is_continuous:
    #     latent_dist_cont = [torch.cat([x[0] for x in latent_dist_list['cont']], dim=0),
    #                         torch.cat([x[1] for x in latent_dist_list['cont']], dim=0)]
    # else:
    #     latent_dist_cont = []
    # if model.is_discrete:
    #     latent_dist_disc = [torch.cat([x[i] for x in latent_dist_list['disc']], dim=0) \
    #                         for i in range(len(latent_dist_list['disc'][0]))]
    # else:
    #     latent_dist_disc = []
    # latent_dist = {'cont':latent_dist_cont, 'disc':latent_dist_disc}
    # return recon_sample, latent_dist


def embed_unicoord_in_adata(adata, adata_ref = None, 
                            keep_dims = None, chunk_size = 20000,
                            only_unsup = False, 
                            only_sup = False, obsm_name = 'unicoord'):
    '''
    find posterior for every cell and the mean of continuous dimensions will be in the obsm of adata
    if only_unsup, unsupervised continuous dimentisons will be kept
    if only_sup, supervised  continuous dimentisons will be kept
    only_unsup and only_sup cannot be both True
    ''' 
    if not adata_ref:
        adata_ref = adata
    unc_stuffs = adata_ref.uns['unc_stuffs']

    cells_used = adata.obs_names
    cells_chunk = [list(cells_used[i:i+chunk_size]) \
                   for i in range(0,len(cells_used),chunk_size)]
    all_data = _tensor_from_adata(adata, cells_chunk, unc_stuffs['genes_used'], 
                                  unc_stuffs['slot'],label_needed=False)
    def grep_embed(data, keep_dims = keep_dims):
        recon_sample, latent_dist = _find_posterior(data, unc_stuffs['model'])
        if not keep_dims:
            keep_dims = list(range(latent_dist['cont'][0].shape[1]))

        if only_unsup and only_sup:
            raise Exception("only_unsup and only_sup cannot be both True")
        if only_sup:
            # the first n_cont_sup cont dims are supervised 
            embed = latent_dist['cont'][0][:,:unc_stuffs['n_dims']['cont']['n_cont_sup']]
        elif only_unsup:
            embed = latent_dist['cont'][0][:,unc_stuffs['n_dims']['cont']['n_cont_sup']:]
        else:
            embed = latent_dist['cont'][0][:,keep_dims]
        return np.array(embed.detach().cpu()).astype('float32')

    adata.obsm[obsm_name] = np.concatenate([grep_embed(data) for data in all_data], axis=0)

def predcit_unicoord_in_adata(adata, adata_ref = None, chunk_size = 20000,
                                evaluate = False, show_comparison = False):
    if not adata_ref:
        adata_ref = adata
    unc_stuffs = adata_ref.uns['unc_stuffs']

    cells_used = adata.obs_names
    cells_chunk = [list(cells_used[i:i+chunk_size]) \
                   for i in range(0,len(cells_used),chunk_size)]
    all_data = _tensor_from_adata(adata, cells_chunk, unc_stuffs['genes_used'], 
                                  unc_stuffs['slot'],label_needed=False)
    def grep_prediction(data):
        recon_sample, latent_dist = _find_posterior(data, unc_stuffs['model'])
        if 'cont' in latent_dist:
            n_dim_cont = unc_stuffs['n_dims']['cont']
            predicted_cont = latent_dist['cont'][0][:,:n_dim_cont['n_cont_sup'] + n_dim_cont['n_diff']]
            predicted_cont = pd.DataFrame(np.array(predicted_cont.detach().cpu()))
            predicted_cont.columns = [s+'_unc_infered' for s in unc_stuffs['obs_fitting']['cont']] + \
                                    ['unc_diffusion_'+str(i) for i in range(n_dim_cont['n_diff'])]
            predicted_cont.index = range(predicted_cont.shape[0])
        else:
            predicted_cont = pd.DataFrame()

        if 'disc' in latent_dist:
            n_dim_disc = unc_stuffs['n_dims']['disc']
            predicted_disc = latent_dist['disc'][:len(n_dim_disc['n_disc_sup'])+len(n_dim_disc['n_clus'])]
            predicted_disc = [np.array(torch.argmax(t.detach().cpu(),dim = 1)) for t in predicted_disc]
            disc_mapping = unc_stuffs['disc_mapping']
            disc_map_back = {obs: {v:k for k,v in disc_mapping[obs].items()} for obs in disc_mapping}
            for obs in disc_map_back:
                disc_map_back[obs][0] = 'rare types'

            for idx,disc in enumerate(disc_map_back.keys()):
                predicted_disc[idx] = [disc_map_back[disc][i] for i in predicted_disc[idx]]
            predicted_disc = pd.DataFrame(predicted_disc).T
            predicted_disc.columns = [s+'_unc_infered' for s in list(unc_stuffs['obs_fitting']['disc'].keys())] + \
                                    ['unc_clusters_'+str(i) for i in range(len(n_dim_disc['n_clus']))]
            predicted_disc.index = range(predicted_disc.shape[0])
        else:
            predicted_disc = pd.DataFrame()
   
        return pd.concat([predicted_cont, predicted_disc], axis = 1)

    # return [grep_prediction(data) for data in all_data]
    obs = pd.concat([grep_prediction(data) for data in all_data], axis=0)
    obs.index = adata.obs_names
    for c in obs.columns:
        adata.obs[c] = obs[c]
    # adata.obs = pd.concat([adata.obs,obs], axis = 1)
    if show_comparison:
        colors = list(chain.from_iterable([[s, s+'_unc_infered'] \
                      for s in unc_stuffs['obs_fitting']['cont'] + \
                                 list(unc_stuffs['obs_fitting']['disc'].keys())]))
        sc.pl.embedding(adata, 'X_umap',legend_loc='on data', legend_fontsize=10,
                        color=colors, ncols=2)

def generate_unicoord_in_adata(adata, adata_ref = None, chunk_size = 20000,
                                cells_used = None, set_value = None):
    if not adata_ref:
        adata_ref = adata
    unc_stuffs = adata_ref.uns['unc_stuffs']
    if not cells_used:
        cells_used = adata.obs_names
    cells_chunk = [list(cells_used[i:i+chunk_size]) \
                   for i in range(0,len(cells_used),chunk_size)]
    all_data = _tensor_from_adata(adata, cells_chunk, unc_stuffs['genes_used'], 
                                  unc_stuffs['slot'],label_needed=False)
    if set_value is not None:
        set_chunk_value = {}
        for k,v in set_value.items():
            if hasattr(v, "__len__") and not isinstance(v,str):
                set_chunk_value[k] = [list(v[i:i+chunk_size]) \
                                    for i in range(0,len(v),chunk_size)]
            else:
                set_chunk_value[k] = v
        set_chunk_value = [{k:v if not hasattr(v, "__len__") or isinstance(v,str) else v[i] \
                            for k,v in set_chunk_value.items()} \
                        for i in range(len(cells_chunk))]
    else:
        set_chunk_value = [None] * len(cells_chunk)
                    
    def generate_data(data, set_chunk):
        recon_sample, latent_dist = _find_posterior(data, unc_stuffs['model'])
        if not set_chunk:
            return np.array(recon_sample.detach().cpu()).astype('float32')
        model = unc_stuffs['model']
        for k,v in set_chunk.items():
            if k in unc_stuffs['obs_fitting']['cont']:
                if hasattr(v, "__len__") and not isinstance(v,str):
                    v = torch.FloatTensor(v)
                    if gpu:
                        v = v.cuda()
                latent_dist['cont'][0][:, [i for i,c in \
                                        enumerate(unc_stuffs['obs_fitting']['cont'])\
                                        if c == k][0]] = v
            elif k in unc_stuffs['obs_fitting']['disc']:
                v = unc_stuffs['disc_mapping'][k][v]
                pos = [i for i,c in \
                        enumerate(unc_stuffs['obs_fitting']['disc'])\
                        if c == k][0]
                latent_dist['disc'][pos][:,:] = 0
                latent_dist['disc'][pos][:,v] = 1
            else:
                raise ValueError('value to set %s not exist in the model'%(k))
        with torch.no_grad():
            latent_sample = model.reparameterize(latent_dist)
            # return latent_sample
            return np.array(model.decode(latent_sample).detach().cpu()).astype('float32')

    # return [generate_data(data) for data in all_data]
    mtx = np.concatenate([generate_data(data,set_chunk) for data,set_chunk in zip(all_data, set_chunk_value) ], axis=0)
    obs = adata[cells_used,:].obs
    gen_adata = sc.AnnData(X = mtx, obs = obs, 
                            var = pd.DataFrame(index = unc_stuffs['genes_used'], 
                                                columns=['features'], data=unc_stuffs['genes_used']))
    gen_adata.obs_names = list(map(lambda x:x+'_gen', gen_adata.obs_names))
    return gen_adata

def _create_model_from_stuffs(adata):
    unc_stuffs = adata.uns['unc_stuffs']
    genes_used = unc_stuffs['genes_used']
    latent_spec = {'cont': sum(unc_stuffs['n_dims']['cont'].values()), 
                   'disc': unc_stuffs['n_dims']['disc']['n_disc_sup'] + \
                               unc_stuffs['n_dims']['disc']['n_clus'] + \
                               unc_stuffs['n_dims']['disc']['n_disc']}
    if not latent_spec['cont']: del latent_spec['cont']
    if not latent_spec['disc']: del latent_spec['disc']

    model = VAE(latent_spec=latent_spec, 
                data_size = (len(genes_used),1), 
                use_cuda = gpu)
    model.training = False

    state_dict_np = unc_stuffs['model_parameters']
    state_dict_pt = OrderedDict()
    for k,v in state_dict_np.items():
        state_dict_pt[k] = torch.FloatTensor(v)
        if gpu:
            state_dict_pt[k] = state_dict_pt[k].cuda()
    model.load_state_dict(state_dict_pt)
    del unc_stuffs['model_parameters']
    return model

def write_scu_h5ad(adata, file_name, only_model = False):
    tmp = sc.AnnData(var = adata.var.iloc[:2,:], 
        obs = adata.obs.iloc[:2,:], 
        X = np.zeros((2,2)))
    tmp.uns['unc_stuffs'] = adata.uns['unc_stuffs']
    tmp = tmp.copy()
    unc_stuffs = tmp.uns['unc_stuffs']

    # save model parameters as dict of np.array, which can be saved in h5ad
    model = unc_stuffs['model']
    state_dict = model.state_dict()
    state_dict_np = OrderedDict()
    for k,v in state_dict.items():
        state_dict_np[k] = np.array(v.detach().cpu())
    unc_stuffs['model_parameters'] = state_dict_np
    del unc_stuffs['model']

    # dict in h5ad may raise error if some keys contains "/", change it to dataframe
    # if it's in obs.columns, change its name!
    unc_stuffs['disc_mapping'] = {k:np.array(list(v.items())) \
                                    for k,v in unc_stuffs['disc_mapping'].items()}
    unc_stuffs['disc_order'] = list(unc_stuffs['disc_mapping'].keys())
    unc_stuffs['obs_fitting']['disc'] = np.array(list(unc_stuffs['obs_fitting']['disc'].items()))


    if only_model:
        tmp.write_h5ad(file_name)
    else:
        orig_unc = adata['unc_stuffs']
        adata['unc_stuffs'] = unc_stuffs
        adata.write_h5ad(file_name)
        adata['unc_stuffs'] = orig_unc


def load_scu_h5ad(file_name):

    adata = sc.read_h5ad(file_name)

    unc_stuffs = adata.uns['unc_stuffs']

    # transfer back array things
    unc_stuffs['disc_mapping'] = {k:{c:int(i) for c,i in dict(unc_stuffs['disc_mapping'][k]).items()}\
                                     for k in unc_stuffs['disc_order']}
    del unc_stuffs['disc_order']

    unc_stuffs['obs_fitting']['disc'] = {k:int(v) for k,v in dict(unc_stuffs['obs_fitting']['disc']).items()}
    unc_stuffs['obs_fitting']['cont'] = list(unc_stuffs['obs_fitting']['cont'])

    unc_stuffs['genes_used'] = list(unc_stuffs['genes_used'])
    unc_stuffs['loss_weights']['cont_capacity'] = list(unc_stuffs['loss_weights']['cont_capacity'])
    unc_stuffs['loss_weights']['disc_capacity'] = list(unc_stuffs['loss_weights']['disc_capacity'])
    for k in ['dim_definition','loss']:
        unc_stuffs[k] = {k:list(v) for k,v in unc_stuffs[k].items()}

    unc_stuffs['n_dims']['disc'] = {k:list(v) for k,v in unc_stuffs['n_dims']['disc'].items()}

    # create the model and set parameters
    

    unc_stuffs['model'] = _create_model_from_stuffs(adata)
    
    return adata