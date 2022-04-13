import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def draw_scatter_with_discrete_labels(embeddings, labels):
    '''
    embeddings: np.array, (N,2),
        Coordination of data
    labels: pd.DataFrame, (N,D)
        Lables used to color dots
        D is the number of labels, each label will generate a subplot
    '''
    ncol = 2
    nrow = len(labels.columns)//2+1
    fig = plt.figure(figsize=(16, 8*nrow), dpi= 100)
    for idx,l in enumerate(labels.columns):
        label = labels[l]
        ax = fig.add_subplot(nrow,ncol,idx+1)
        for i in np.unique(label):
            index = label == i
            ax.scatter(embeddings[index,0],embeddings[index,1],label=i,s = 5)
            ax.legend(fancybox=True, framealpha=0.5, prop={'size': 15})
            ax.title.set_text(l)
            ax.title.set_size(fontsize=20)
    return fig

def draw_scatter_with_continuous_labels(embeddings, labels):
    '''
    embeddings: np.array, (N,2),
        Coordination of data
    labels: pd.DataFrame, (N,D)
        Lables used to color dots
        D is the number of labels, each label will generate a subplot
    '''
    ncol = 2
    nrow = len(labels.columns)//2+1
    fig = plt.figure(figsize=(16, 8*nrow), dpi= 100)
    for idx,l in enumerate(labels.columns):
        label = labels[l]
        ax = fig.add_subplot(nrow,ncol,idx+1)
        sc = ax.scatter(embeddings[:,0],embeddings[:,1], c=label, s = 5, cmap='Spectral_r')
        ax.title.set_text(l)
        ax.title.set_size(fontsize=20)
        fig.colorbar(sc, ax=ax)
    return fig
        
def draw_loss_curves(losses):
    '''
    losses: dict
        {loss_name:list[float]}, loss saved during training steps
    '''
    df = pd.DataFrame(losses)
    loss_used = []
    for loss in df:
        if loss.startswith('kl_loss_cont_') or loss.startswith('kl_loss_disc_'):
            continue
        loss_used.append(loss)
    ncol = 2
    nrow = len(loss_used)//2+1
    fig = plt.figure(figsize=(10, 5*nrow), dpi= 100)
    for idx, loss in enumerate(loss_used):
        ax = fig.add_subplot(nrow,ncol,idx+1)
        ax.plot(df[loss], label = loss)
        ax.title.set_text(loss)
    return fig