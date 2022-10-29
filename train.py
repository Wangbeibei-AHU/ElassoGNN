import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import accuracy, proximal_op
import scipy.io as sio
import warnings
import pdb
import math

class ELassoGNN:
    
    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 100
        self.best_graph = None
        self.weights = None
        self.train_loss = 0
        self.estimator = None
        self.model = model.to(device)
       

    def fit(self, features, adj, labels, idx_train, idx_val, **kwargs):
        args = self.args
        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        estimator = EstimateAdj(adj, symmetric=args.symmetric, device=self.device).to(self.device)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(),
                              momentum=0.9, lr=args.lr_adj)

        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            for i in range(int(args.outer_steps)):
                self.train_adj(epoch, features, adj, labels,
                        idx_train, idx_val)
            for i in range(int(args.inner_steps)):
                self.train_gcn(epoch, features, estimator.estimated_adj,
                         labels, idx_train, idx_val)
        
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(args)

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)

    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val):
        args = self.args
        estimator = self.estimator
        adj_nor = estimator.normalize()
        
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, adj_nor)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(features, adj_nor)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
    
        #-----------------
        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj_nor.detach()
            self.train_loss = loss_train
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj_nor.detach()
            self.train_loss = loss_train
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())


    def train_adj(self, epoch, features, adj, labels, idx_train, idx_val):
        estimator = self.estimator
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()
        
        normalized_adj = estimator.normalize()
       
        output = self.model(features, normalized_adj)
        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_gcn.backward()

        self.optimizer_adj.step()
        
        # -----------------proximal operator-----------------------
        data = proximal_op(adj, estimator.estimated_adj, args.beta)
        estimator.estimated_adj.data.copy_(data) 
        #------------------------------

        self.model.eval()
        normalized_adj = estimator.normalize()
        
        output = self.model(features, normalized_adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        
        if epoch % 100 == 0:
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t))
        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.train_loss = loss_gcn
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = normalized_adj.detach()
            self.train_loss = loss_gcn
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())



    def test(self, features, labels, idx_val, idx_test):
        """Evaluate the performance of ProGNN on test set
        """
        print("\t=== testing ===")
        self.model.eval()
        adj = self.best_graph
        
        if self.best_graph is None:
            best_adj = self.estimator.normalize()
            
        output = self.model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        loss_vall = F.nll_loss(output[idx_val], labels[idx_val])
        print("Test set results:",
              "loss_test= {:.4f}".format(loss_test.item()),
              "loss_vall= {:.4f}".format(loss_vall.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()
       

class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        self.estimated_adj = nn.Parameter(torch.FloatTensor(adj.shape[0], adj.shape[1]))
        self._init_estimation(adj)
        self.device = device
        self.ori = adj
        

    def _init_estimation(self, adj):
        with torch.no_grad():
            self.estimated_adj.data.copy_(adj)
       
    def forward(self):
        
        return self.estimated_adj

    def normalize(self):# M, ad  
        adj = self.estimated_adj* self.ori
        normalized_adj = self._normalize(adj+torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj


    def _normalize(self, mx):
        rowsum = mx.sum(1)#+1e-8
        r_inv = rowsum.pow(-1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        return mx
