import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import pdb


def load_noisy_data(p, n, dataset="cora",device='gpu', path="./data/"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    path = path + dataset
    print(path)
    if dataset == 'cora':        # cora
        noisy_ratio=[0.0025, 0.0035, 0.0045]
        ratio = noisy_ratio[p]
        adj = sio.loadmat('./data/cora/dense_graph/adj-'+str(ratio))
        adj = adj['adj']
        #load_feature
        feat = sio.loadmat(path + '/feature.mat')
        features = feat['matrix']
        labels = sio.loadmat(path + '/label.mat')
        labels = labels['matrix']
        labels = np.where(labels)[1]

        #dense_edge_idx
        idx = np.loadtxt("./data/cora/random_split/"+'idx'+str(n),dtype='int')
        idx_train = idx[0:140]
        idx_val = idx[140:640]
        idx_test = idx[640:1640]
    elif dataset == 'citeseer':        # citeseer
        noisy_ratio=[0.0015, 0.002, 0.0025]
        ratio = noisy_ratio[p]
        # load_dense_edge
        adj = sio.loadmat('./data/citeseer/dense_graph/adj-'+str(ratio))
        adj = adj['adj']

        #load_feature
        feat = sio.loadmat(path + '/feature.mat')
        features = feat['matrix']
        labels = sio.loadmat(path + '/label.mat')
        labels = labels['matrix']
        labels = np.where(labels)[1]

        #dense_edge_idx
        idx = np.loadtxt("./data/citeseer/random_split/"+'idx'+str(n),dtype='int')
        idx_train = idx[0:120]
        idx_val = idx[120:620]
        idx_test = idx[620:1620]
    elif dataset == 'amazon_electronics_photo':        # citeseer
        # load_original_edge
        # adj = sio.loadmat(path + '/adj.mat')
        # adj = adj['adj']

        # load_dense_edge
        adj = sio.loadmat('./data/amazon_electronics_photo/adj-0.006.mat')
        adj = adj['adj']
        feat = sio.loadmat(path + '/features.mat')
        features = feat['feat']
        labels = sio.loadmat(path + '/labels.mat')
        labels = labels['label'].flatten()
        idx = np.loadtxt(path+'/20_30_10/idx'+str(n),dtype='int')
        idx_train = idx[0:160]
        idx_val = idx[160:400]
        idx_test = idx[400:]
    elif dataset == 'amazon_electronics_computers':        # citeseer
        adj = sio.loadmat('./data/amazon_electronics_computers/adj-0.006.mat')
        adj = adj['adj']
        feat = sio.loadmat(path + '/features.mat')
        features = feat['feat']
        labels = sio.loadmat(path + '/labels.mat')
        labels = labels['label'].flatten()
        idx = np.loadtxt(path+'/20_30_10/idx'+str(n),dtype='int')
        idx_train = idx[0:200]
        idx_val = idx[200:500]
        idx_test = idx[500:]
    
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    labels = torch.LongTensor(labels)
    adj = torch.FloatTensor(np.array(adj.todense()))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj.to(device),  features.to(device), labels.to(device), idx_train.to(device), idx_val.to(device), idx_test.to(device)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(abs(mx).sum(1))#+1e-8
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_norm(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.linalg.norm(mx.todense(),ord=2,axis=1)
    r_inv = np.power(rowsum+1e-15, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # pdb.set_trace()
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def denseTosparse(mx,index):
    indices_row = index[0,:]
    indices_col = index[1,:]
    values = mx[indices_row,indices_col]
    del indices_row
    del indices_col
    return values
    

def sparseTodense(mx,values,index):
    # pdb.set_trace()
    zero_v = torch.zeros_like(mx)
    indices_row = index[0,:]
    indices_col = index[1,:]
    zero_v[indices_row,indices_col] = values
    # del zero_v
    del indices_row
    del indices_col
    # pdb.set_trace()
    return zero_v


def proximal_op(adj, estimated_adj, beta):
    index = adj.nonzero().t()
    zero_vec = torch.zeros_like(adj)
    Z = torch.where(adj==0, zero_vec, estimated_adj)
    Z = torch.where(Z<0, zero_vec, Z)
    
    Z_values = denseTosparse(Z,index)
    data = adj
    data_values = denseTosparse(data,index)
    for i in range(50):
        row_sum = torch.sum(sparseTodense(adj,data_values,index),1)*beta
        data_values_addrowsum = row_sum[index[0,:][torch.arange(index.shape[1])]]+data_values[torch.arange(index.shape[1])]
        data_values = data_values*(Z_values/(data_values_addrowsum+1e-8))
    # ----------------normalizition------------
    data_values = data_values/data_values.max()
    data = sparseTodense(adj,data_values,index)

    return data