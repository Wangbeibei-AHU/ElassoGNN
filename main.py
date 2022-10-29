import time
import argparse
import numpy as np
import torch
import pdb
import os.path as osp
import os 
from gcn import GCN
from train import ELassoGNN
from utils import load_noisy_data


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true', default=False, help='test the performance of gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='amazon_electronics_computers', choices=['cora',  'citeseer','amazon_electronics_photo','amazon_electronics_computers'], help='dataset')
parser.add_argument('--epochs', type=int,  default=1000, help='Number of epochs to train.')
parser.add_argument('--beta', type=float, default=0.0005, help='weight of nuclear norm')
parser.add_argument('--inner_steps', type=int, default=5, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.02, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False,
            help='whether use symmetric matrix')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
        torch.cuda.manual_seed(args.seed)

def mainf(p, n):
        adj,  features, labels, idx_train, idx_val, idx_test = load_noisy_data(p, n, args.dataset,device=device)

        model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,device=device)#, device=device

        elassognn = ELassoGNN(model, args, device)
        elassognn.fit(features, adj, labels, idx_train, idx_val)

        result = elassognn.test(features, labels, idx_val, idx_test)
        return result

num = np.zeros([3,10])
for p in range(3):
        for i in range(10):
            num[p][i]= mainf(p, i)
print("result_train: ", np.mean(num[0]), "+", np.std(num[0]))
print("result_train: ", np.mean(num[1]), "+", np.std(num[1]))
print("result_train: ", np.mean(num[2]), "+", np.std(num[2]))
