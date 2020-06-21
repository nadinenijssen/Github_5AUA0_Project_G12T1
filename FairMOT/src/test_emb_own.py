from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import argparse
import torch
import json
import time
import os
import cv2
import math

from sklearn import metrics
from scipy import interpolate
import numpy as np
from torchvision.transforms import transforms as T
import torch.nn.functional as F
from models.model import create_model, load_model
from datasets.dataset.jde import JointDataset, collate_fn
from models.utils import _tranpose_and_gather_feat
from utils.utils import xywh2xyxy, ap_per_class, bbox_iou
from opts import opts
from models.decode import mot_decode
from utils.post_process import ctdet_post_process

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
plt.rcParams["figure.figsize"] = (10,10) #nice big plots
import numpy as np
from torchvision.transforms import ToPILImage

from utils.image import gaussian_radius

def indtoxy(ind, ratio=4, input_w=1088, input_h=608):
    if ind != 0:
        w_out, h_out = divmod(ind,input_w//ratio)
        #print(f"out: ({w_out},{h_out})")
        return w_out*ratio, h_out*ratio
    else:
        return None

def wh_decode(wh_tensor, ratio=4):
    wh_numpy = wh_tensor.cpu().numpy()
    wh_numpy_decode = np.rint(ratio*wh_numpy)
    return wh_numpy_decode[0], wh_numpy_decode[1]
    
    
    
def save_TPR(TPR, far_levels):
    TPR_dict = {'TPR@FAR={:.7f}'.format(k): [TPR[v]] for v, k in enumerate(far_levels)}
    TPR_dataframe = pd.DataFrame(data=TPR_dict)
    TPR_exp_name = '_'.join(['TPR', opt.arch, opt.exp_id])
    TPR_filename = '.'.join([TPR_exp_name, 'xlsx'])
    TPR_path = os.path.join('../results/embeddings', TPR_filename)
    writer = pd.ExcelWriter(TPR_path)
    TPR_dataframe.to_excel(writer)
    writer.save()
    
def plot_embeddings(embeddings, id_labels):
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
        # legend=False,     #legend="full",
        legend="full",
        # alpha=0.3
    )
    # Save plot
    plot_exp_name = '_'.join(['embeddings', opt.arch, opt.exp_id])
    plot_filename = '.'.join([plot_exp_name, 'png'])
    plot_path = os.path.join('../results/embeddings', plot_filename)
    plt.savefig(plot_path)


def test_emb(
        opt,
        batch_size=16,
        img_size=(1088, 608),
        print_interval=40,
):
#     data_cfg = opt.data_cfg
#     f = open(data_cfg)
#     data_cfg_dict = json.load(f)
#     f.close()
#     nC = 1
#     test_paths = data_cfg_dict['test_emb']
#     dataset_root = data_cfg_dict['root']

    test_paths = {"mot17": opt.test_emb_data}
    dataset_root = opt.data_dir
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    # model = torch.nn.DataParallel(model)
    model = model.to(opt.device)
    model.eval()

    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(opt, dataset_root, test_paths, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                             num_workers=8, drop_last=False)
    embedding, id_labels = [], []
    print('Extracting pedestrain features...')
    for batch_i, batch in enumerate(dataloader):
        t = time.time()
        
        # print(batch.keys())
        img_tensor = batch['input'][0,:,:,:] 
        #img  = batch['input'][0,:,:,:].cpu().numpy().reshape(608,1088,3)
        to_pil = ToPILImage()
        img_pil = to_pil(img_tensor)
        img_np =  np.array(img_pil)


        patcheslist = []
        radiuslist = []
        #plot bounding boxes
        for i in range(128):
            if batch['reg_mask'][0,i].cpu().numpy():
                y,x = indtoxy(batch['ind'][0,i].cpu().numpy())
                # print(batch['wh'][0,i])
                w,h = wh_decode(batch['wh'][0,i])
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radiuslist.append(radius)
                #print(batch['ind'][0,i])
                #print(indtoxy(batch['ind'][0,i].cpu().numpy()))
                #w, h = indtoxy(batch['ind'][0,i].cpu().numpy())
                # print(f"x: {x}, y: {y}")
                # print(f"w: {w}, h: {h}")
                # print(f"radius: {radius}")
                x_plt = x - 0.5*w   # left coordinate
                y_plt = y - 0.5*h # bottom coordinate
                # print(f"x_plt: {x_plt}, y_plt: {y_plt}")
                patcheslist.append(patches.Rectangle((x_plt,y_plt),w,h,linewidth=1,edgecolor='r',facecolor='none'))
        
        n_lbb = min(len(radiuslist), 10)
        sort_index = np.argsort(radiuslist)[::-1][:n_lbb]

        patcheslist_lbb = []
        for i in range(n_lbb):
            # print(radiuslist[sort_index[i]])
            patcheslist_lbb.append(patcheslist[sort_index[i]])

        # if batch_i == 0 or batch_i == 9:
        #     # Create figure and axes
        #     fig, (ax1,ax2) = plt.subplots(2)
    
        #     colors = 100*np.random.rand(len(patcheslist_lbb))
        #     p = PatchCollection(patcheslist_lbb, alpha=0.4)
        #     p.set_array(np.array(colors))
        #     ax1.add_collection(p)
    
    
        #     # Display the image
        #     ax1.imshow(img_np)
    
        #     # Create a Rectangle patch
        #     #rect = patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none')
    
        #     # Add the patch to the Axes
        #     #ax1.add_patch(rect)
        #     # print(batch['hm'].shape)
        #     ax2.imshow(batch['hm'][0,0,:,:])
        #     # Save plot
        #     plot_exp_name = '_'.join(['bb{}'.format(batch_i), opt.arch, opt.exp_id])
        #     plot_filename = '.'.join([plot_exp_name, 'png'])
        #     plot_path = os.path.join('/content/gdrive/My Drive/5AUA0_Project_Group12_Team1/Github_5AUA0_Project_G12T1/FairMOT/results/embeddings', plot_filename)
        #     plt.savefig(plot_path)
        
        output = model(batch['input'].cuda())[-1]
        id_head = _tranpose_and_gather_feat(output['id'], batch['ind'].cuda())
        id_head = id_head[batch['reg_mask'].cuda() > 0].contiguous()
        emb_scale = math.sqrt(2) * math.log(opt.nID - 1)
        id_head = emb_scale * F.normalize(id_head)
        id_target = batch['ids'].cuda()[batch['reg_mask'].cuda() > 0]
        
        # id_head_lbb = []
        # id_target_lbb = []
        # for i in range(n_lbb):
        #     id_head_lbb.append(id_head[sort_index[i]].unsqueeze(0))
        #     id_target_lbb.append(id_target[sort_index[i]].unsqueeze(0))
        # id_head_lbb = torch.cat(id_head_lbb) # list to tensor
        # id_target_lbb = torch.cat(id_target_lbb) # list to tensor
        
        # # for i in range(0, id_head.shape[0]):
        #     # if len(id_head.shape) == 0:
        #         # continue
        #     # else:
        #         # feat, label = id_head[i], id_target[i].long()
        #     # if label != -1:
        #         # embedding.append(feat)
        #         # id_labels.append(label)
        # for i in range(0, id_head_lbb.shape[0]):
        #     if len(id_head_lbb.shape) == 0:
        #         continue
        #     else:
        #         feat, label = id_head_lbb[i], id_target_lbb[i].long()
        #     if label != -1:
        #         embedding.append(feat)
        #         id_labels.append(label)
        
        if batch_i == 0:
            id_target_lbb = []
            for i in range(n_lbb):
                id_target_lbb.append(id_target[sort_index[i]].unsqueeze(0))
            # id_target_lbb = torch.cat(id_target_lbb) # list to tensor
        
        for i in range(0, id_head.shape[0]):
            if len(id_head.shape) == 0:
                continue
            else:
                feat, label = id_head[i], id_target[i].long()
            if label != -1:
                if label in id_target_lbb:
                    embedding.append(feat)
                    id_labels.append(label)        
        
        if batch_i % print_interval == 0:
            print(
                'Extracting {}/{}, # of instances {}, time {:.2f} sec.'.format(batch_i, len(dataloader), len(id_labels),
                                                                               time.time() - t))
                                                                               
        if batch_i == 99:
            break                                                          
        
    print('Computing pairwise similairity...')
    if len(embedding) < 1:
        return None
    embedding = torch.stack(embedding, dim=0).cuda()
    id_labels = torch.LongTensor(id_labels)
    n = len(id_labels)
    print(n, len(embedding))
    assert len(embedding) == n
    
    # make embeddings visualization
    plot_embeddings(embedding, id_labels)
    
    embedding = F.normalize(embedding, dim=1)
    pdist = torch.mm(embedding, embedding.t()).cpu().numpy()
    gt = id_labels.expand(n, n).eq(id_labels.expand(n, n).t()).numpy()

    up_triangle = np.where(np.triu(pdist) - np.eye(n) * pdist != 0)
    pdist = pdist[up_triangle]
    gt = gt[up_triangle]

    far_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    far, tar, threshold = metrics.roc_curve(gt, pdist)
    interp = interpolate.interp1d(far, tar)
    tar_at_far = [interp(x) for x in far_levels]
    for f, fa in enumerate(far_levels):
        print('TPR@FAR={:.7f}: {:.4f}'.format(fa, tar_at_far[f]))
        
    # save TPR (own code)
    save_TPR(tar_at_far, far_levels)

    return tar_at_far

if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    print("gpus", ",".join(map(str, opt.gpus)))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, opt.gpus))
    with torch.no_grad():
        map = test_emb(opt, batch_size=4)
