UniCoord (Universal Coordination system for scRNA-seq) is a deep learning based method for embedding, annotating and generating single cell RNA sequencing data. Read our [preprint biorxiv paper](https://www.biorxiv.org/content/10.1101/2021.09.09.459281v1) for detail information.

- [Requirements](#requirements)
- [Import UniCoord](#import-unicoord)
- [Annotate your data](#annotate-your-data)
- [Train your own annotation model](#train-your-own-annotation-model)
- [Get UniCoord embedding](#get-unicoord-embedding)
- [Generate user defined cells](#generate-user-defined-cells)
## Requirements

UniCoord requires python>=3.7, torch, scanpy

## Import UniCoord

```bash
# in bash
git clone git@github.com:pluto-the-lost/unicoord.git
```

```python
# in python, better in jupyter notebook
import sys
sys.path.append('path/to/unicoord/folder')
from unicoord import scu
```

## Annotate your data

You need a trained UniCoord model to do annotation, you can download our pretrained model [here](https://cloud.tsinghua.edu.cn/d/13021decce6c40ad9c4e/). Detail information about pretrained models are in readme.csv in the link.

Or you can also train your own model, find tutorial [here](#train-your-own-annotation-model).

Here we assume that you already have a model file.

```python
import scanpy as sc

# load your dataset
adata = sc.read_h5ad('your h5ad file')
# Normalize, UniCoord expect log1p(TP10k) data
adata = adata.raw.to_adata()
sc.pp.normalize_total(adata, target_sum=1e4 ,exclude_highly_expressed= True)
sc.pp.log1p(adata)

# load UniCoord model
model = scu.load_scu_h5ad('./pretrained_models/unc_model_TBMU.h5ad')

# do prediction
scu.predcit_unicoord_in_adata(adata, ref = model)
```

Now your adata will be added some obs columns whose names end with '_unc_infered'. The meaning of infered annotations depends on annotations on which the model was trained. For our pretrained model, they are usually different level of cell types, functional score of some pathways, etc. 

## Train your own annotation model

```python
# adata should be your dataset with annotation
# here we use Tabula Muris as example
adata = sc.read_h5ad('./tabularMuris/TBMU.h5ad')

# Normalization, UniCoord expect log1p(TP10k) data
adata = adata.raw.to_adata()
sc.pp.normalize_total(adata, target_sum=1e4 ,exclude_highly_expressed= True)
sc.pp.log1p(adata)

# build a model, specify the columns to be learned, 
# can be any column in the adata.obs dataframe
scu.model_unicoord_in_adata(adata, 
                            n_cont=50, n_diff=0, n_clus = [],
                            obs_fitting=['cell_ontology_class',
                                         'free_annotation','mouse.id','mouse.sex',
                                         'tissue','tissue_tSNE_1','tissue_tSNE_2','seq_tech'], 
                            min_obs = 500)

# train the model
scu.train_unicoord_in_adata(adata, epochs=100, chunk_size=20000, slot = "cur")

# use the model to predict another dataset, 
# expression matrix in bdata should also be log1p(TP10k)
scu.predcit_unicoord_in_adata(bdata, ref = adata)

# save the model with the training dataset
scu.write_scu_h5ad(adata, './tabularMuris/TBMU.h5ad')

# load the model with training dataset
adata = scu.load_scu_h5ad('./tabularMuris/TBMU.h5ad')

# save the model only, without data
scu.write_scu_h5ad(adata, './pretrained_models/unc_model_TBMU.h5ad', only_model=True)

# load the model
model = scu.load_scu_h5ad('./pretrained_models/unc_model_TBMU.h5ad')
```

## Get UniCoord embedding

UniCoord embeds expression matrix to a low-dimensional latent space, whose dimensionalities are composed of supervised part and unsupervised part. The disentanglement feature of VAE gives UniCoord the capability to remove some unwanted attributes, like batch index, and only use the remaining information to do downstream analysis.

```python
# adata should be your dataset with annotation
# here we use Tabula Muris as example
adata = sc.read_h5ad('tabularMuris/TBMU.h5ad')

# Normalize, UniCoord expect log1p(TP10k) data
adata = adata.raw.to_adata()
sc.pp.normalize_total(adata, target_sum=1e4 ,exclude_highly_expressed= True)
sc.pp.log1p(adata)

# build and train a UniCoord model, 
# what different with training annotation model
# is that embedding model only fit columns you want to remove from data, i.e. batch

scu.model_unicoord_in_adata(adata, n_cont=50, n_diff=0, n_clus = [],
                            obs_fitting=['mouse.id','seq_tech'])

# do embedding, result saved in adata.obsm['unicoord']
scu.embed_unicoord_in_adata(adata, chunk_size=5000)

# use unicoord embedding for downstream analysis 
sc.pp.neighbors(adata, use_rep='unicoord')
sc.tl.leiden(adata, resolution=0.5)
sc.tl.umap(adata)
sc.pl.embedding(adata, 'X_umap', legend_fontsize=10,
                color= ['mouse.id','seq_tech',
                        'tissue','cell_ontology_class'], ncols=1)
```

## Generate user defined cells

UniCoord is a generative model. By setting the latent embedding value, you can have a self-defined dataset. 

```python
# load your dataset as a original data
adata = sc.read_h5ad('your h5ad file')
# Normalize, UniCoord expect log1p(TP10k) data
adata = adata.raw.to_adata()
sc.pp.normalize_total(adata, target_sum=1e4 ,exclude_highly_expressed= True)
sc.pp.log1p(adata)

# load the model
model = scu.load_scu_h5ad('./pretrained_models/unc_model_TBMU.h5ad')

# adata is your dataset
bdata = scu.generate_unicoord_in_adata(adata, ref = model, 
                                       set_value = {'Type':'T cells'})

# set multiple value at the same time
# use an list or array to set cell specific value
cdata = scu.generate_unicoord_in_adata(adata, ref = model, 
                                       set_value = {'Type':'T cells',
                                                    'seq_tech':['Smart-seq2']*1000 + ['10X']*(adata.n_obs-1000)})
```

The output of scu.generate_unicoord_in_adata is a new dataset. For example, bdata will be a new generated AnnData object, containing pseudo-cells whose all attributes equal to cells in adata, while cell type are changed to T cells.

Also you can change other cell attributes, the most recommended changes are batch, seq_tech and trajectory pseudotime.