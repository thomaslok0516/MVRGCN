import numpy as np
import torch.nn.functional as F
from .model.MVRGCN import GCN_ensemble
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, PrePtbDataset
import argparse
import pandas as pd
import time

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
# parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--attack', type=str, default='meta',  help='pertubation rate',choices=['meta','nettack'])
parser.add_argument('--gamma', type=float, default=0.1,  help='selfloop weight')
parser.add_argument('--k', type=float, default=20,  help='knn')
parser.add_argument('--rate', type=float, default=0.05,  help='perturbation rate')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# make sure you use the same data splits as you generated attacks
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data = Dataset(root=r'C:\Users\Administrator\Desktop\dataset', name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test




data_path = r'C:\Users\Administrator\Desktop\dataset'
if args.attack == 'nettack':
    get_perturbed_data = PrePtbDataset(root=data_path,
                                   name=args.dataset,
                                   attack_method=args.attack,
                                   ptb_rate=1.0)
    target_nodes = get_perturbed_data.target_nodes
    
data = Dataset(root=data_path, name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
perturbed_data = PrePtbDataset(root=data_path,
                                name=args.dataset,
                                attack_method=args.attack,
                                ptb_rate=args.rate)
perturbed_adj = perturbed_data.adj
if(args.attack == 'nettack'):
    idx_test = target_nodes
model = GCN_ensemble(nnodes=features.shape[0], nfeat=features.shape[1], nclass=labels.max() + 1,
                                nhid=16, gamma=args.gamma,k=args.k, device=device)

model = model.to(device)
model.fit(features, perturbed_adj, labels, idx_train, idx_val)
model.eval()
output = model.test(idx_test)
print(output)
