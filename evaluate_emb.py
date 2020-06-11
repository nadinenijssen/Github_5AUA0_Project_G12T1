# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:37:13 2020

@author: s166744
"""

import pandas as pd
from sklearn.manifold import TSNE
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# save TPR to excel file
def save_TPR(TPR, far_levels, arch, exp_id):
    TPR_dict = {'TPR@FAR={:.7f}'.format(k): [TPR[v]] for v, k in enumerate(far_levels)}
    TPR_dataframe = pd.DataFrame(data=TPR_dict)
    TPR_exp_name = '_'.join(['TPR', 'MOT17_val', arch, exp_id])
    TPR_filename = '.'.join([TPR_exp_name, 'xlsx'])
    TPR_path = os.path.join('/content/gdrive/My Drive/5AUA0_Project_Group12_Team1/Github_5AUA0_Project_G12T1/FairMOT/results/embeddings', TPR_filename)
    writer = pd.ExcelWriter(TPR_path)
    TPR_dataframe.to_excel(writer)
    writer.save()
    
    
def plot_embeddings(embeddings, id_labels, arch, exp_id):
    # Apply t-SNE to embeddings
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings.cpu())
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    # Plot embeddings
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=id_labels.cpu(),
        palette=sns.color_palette("hls", len(np.unique(id_labels.cpu()))),
        legend=False,     #legend="full",
        alpha=0.3
    )
    # Save plot
    plot_exp_name = '_'.join(['embeddings', 'MOT17_val', arch, exp_id])
    plot_filename = '.'.join([plot_exp_name, 'png'])
    plot_path = os.path.join('/content/gdrive/My Drive/5AUA0_Project_Group12_Team1/Github_5AUA0_Project_G12T1/FairMOT/results/embeddings', plot_filename)
    plt.savefig(plot_path)



#old code
## Print embeddings evaluation results (TPR)
#TPR_filename = os.path.join('/content/gdrive/My Drive/5AUA0_Project_Group12_Team1/Github_5AUA0_Project_G12T1/FairMOT/results/embeddings/TPR_MOT17_val_hrnet_18_hrnet_scratchtrained10BS4_trainval_validation_subset.pt')
#import torch
#TPR = torch.load(TPR_filename)
#far_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
#for f, fa in enumerate(far_levels):
#print('TPR@FAR={:.7f}: {:.4f}'.format(fa, TPR[f]))
    
## save TPR to excel file
#import pandas as pd
#TPR_dict = {'TPR@FAR={:.7f}'.format(k): [TPR[v]] for v, k in enumerate(far_levels)}
#TPR_dataframe = pd.DataFrame(data=TPR_dict)
#writer = pd.ExcelWriter('/content/gdrive/My Drive/5AUA0_Project_Group12_Team1/Github_5AUA0_Project_G12T1/FairMOT/results/embeddings/TPR_test.xlsx')
#TPR_dataframe.to_excel(writer)
#writer.save()
    
## Load embeddings and targets
#embeddings_filename = os.path.join('/content/gdrive/My Drive/5AUA0_Project_Group12_Team1/Github_5AUA0_Project_G12T1/FairMOT/results/embeddings/embeddings_MOT17_val_hrnet_18_hrnet_scratchtrained10BS4_trainval_validation_subset.pt')
#labels_filename = os.path.join('/content/gdrive/My Drive/5AUA0_Project_Group12_Team1/Github_5AUA0_Project_G12T1/FairMOT/results/embeddings/labels_MOT17_val_hrnet_18_hrnet_scratchtrained10BS4_trainval_validation_subset.pt')
#
#import torch
#import numpy as np
#embeddings = torch.load(embeddings_filename)
#id_labels = torch.load(labels_filename)
#print(embeddings.shape)
#print(id_labels.shape)
#len(np.unique(id_labels.cpu()))
    
## Apply t-SNE to embeddings
#from sklearn.manifold import TSNE
#import time
#
#time_start = time.time()
#tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
#tsne_results = tsne.fit_transform(embeddings.cpu())
#print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
## Plot embeddings
## import matplotlib.pyplot as plt
## fig, ax = plt.subplots(1, 1, figsize=(8,8))
## ax.scatter(tsne_results[:,0], tsne_results[:,1])
## ax.set_title('Plot of re-ID features using t-SNE')
## plt.show()
#
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#
#plt.figure(figsize=(16,10))
#sns.scatterplot(
#    x=tsne_results[:,0], y=tsne_results[:,1],
#    hue=id_labels.cpu(),
#    palette=sns.color_palette("hls", len(np.unique(id_labels.cpu()))),
#    legend=False,     #legend="full",
#    alpha=0.3
#)
## please change name according to used model + evaluation dataset
#plt.savefig('/content/gdrive/My Drive/5AUA0_Project_Group12_Team1/Github_5AUA0_Project_G12T1/FairMOT/results/embeddings/embedding_MOT17_val_hrnet_18_hrnet_scratchtrained10BS4_trainval_validation_subset.png')
