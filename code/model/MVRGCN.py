import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
from numba import njit
import torch.optim as optim
from copy import deepcopy
from deeprobust.graph.defense import GraphConvolution
import os
from sklearn.metrics.pairwise import cosine_similarity
from networkx import simrank_similarity,from_scipy_sparse_matrix,from_numpy_matrix

class GCN_ensemble(nn.Module):
    """
    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.
    k : int
        knn

    Examples
    --------
	We can first load dataset and then train GCNJaccard.

    >>> from deeprobust.graph.data import PrePtbDataset, Dataset
    >>> from deeprobust.graph.defense import GCNJaccard
    >>> # load clean graph data
    >>> data = Dataset(root='/tmp/', name='cora', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # load perturbed graph data
    >>> perturbed_data = PrePtbDataset(root='/tmp/', name='cora')
    >>> perturbed_adj = perturbed_data.adj
    >>> # train defense model
    >>> model = GCNJaccard(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu').to('cpu')
    >>> model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=0.03)

    """
    def __init__(self, nnodes,nfeat, nhid, nclass, binary_feature=True, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, gamma=0.1,k=20,bias_init=0,device='cpu'):

        super(GCN_ensemble, self).__init__()
        self.device = device
        self.binary_feature = binary_feature
        self.weight_decay = weight_decay
        self.nclass = nclass
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.gamma = gamma
        self.dropout = dropout
        self.ensemble_coef1 = nn.ParameterList()
        self.ensemble_coef2 = nn.ParameterList()
        self.ensemble_coef1.append(Parameter(torch.FloatTensor(nhid,1)))
        self.ensemble_coef2.append(Parameter(torch.FloatTensor(nhid,1)))
        self.ensemble_bias1 = nn.ParameterList()
        self.ensemble_bias2 = nn.ParameterList()
        self.ensemble_bias1.append(Parameter(torch.FloatTensor(1)))
        self.ensemble_bias2.append(Parameter(torch.FloatTensor(1)))
        self.scores1 = nn.ParameterList()
        self.scores2 = nn.ParameterList()
        self.device = device
        self.k = k
        self.gc11 = GraphConvolution(nfeat,nhid,with_bias=with_bias)
        self.gc12 = GraphConvolution(nhid,nclass,with_bias=with_bias)
        self.gc21 = GraphConvolution(nfeat,nhid,with_bias=with_bias)
        self.gc22 = GraphConvolution(nhid,nclass,with_bias=with_bias)
        self.scores1.append(Parameter(torch.FloatTensor(nfeat,1)))
        self.scores2.append(Parameter(torch.FloatTensor(nfeat,1)))
        for i in range(1):
            self.scores1.append(Parameter(torch.FloatTensor(nhid,1)))
            self.scores2.append(Parameter(torch.FloatTensor(nhid,1)))
        self.bias1 = nn.ParameterList()
        self.bias1.append(Parameter(torch.FloatTensor(1)))
        self.bias2 = nn.ParameterList()
        self.bias2.append(Parameter(torch.FloatTensor(1)))
        self.adj_knn_feature = None
        self.adj_knn_structure = None
        self.D_k1 = nn.ParameterList()
        self.D_k1.append(Parameter(torch.FloatTensor(nfeat,1)))
        self.D_k2 = nn.ParameterList()
        self.D_k2.append(Parameter(torch.FloatTensor(nfeat,1)))
        self.bias_init = bias_init
        self.lr = lr
        for i in range(1):
            self.D_k1.append(Parameter(torch.FloatTensor(nhid,1)))
            self.D_k2.append(Parameter(torch.FloatTensor(nhid,1)))
        self.identity = utils.sparse_mx_to_torch_sparse_tensor(sp.eye(nnodes)).to(device)
        self.D_bias1 = nn.ParameterList()
        self.D_bias1.append(Parameter(torch.FloatTensor(1)))
        self.D_bias2 = nn.ParameterList()
        self.D_bias2.append(Parameter(torch.FloatTensor(1)))
        for i in range(1):
            self.D_bias1.append(Parameter(torch.FloatTensor(1)))
            self.D_bias2.append(Parameter(torch.FloatTensor(1)))
    def initialize(self):
        self.gc11.reset_parameters()
        self.gc12.reset_parameters()
        self.gc21.reset_parameters()
        self.gc22.reset_parameters()
        # 标准化
        for s in self.scores1:
            stdv = 1. / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)
        for s in self.scores2:
            stdv = 1. / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)
        for b in self.bias1:
            b.data.fill_(self.bias_init)
        for b in self.bias2:
            b.data.fill_(self.bias_init)
        # 标准化
        for Dk in self.D_k1:
            stdv = 1. / math.sqrt(Dk.size(1))
            Dk.data.uniform_(-stdv, stdv)
        for Dk in self.D_k2:
            stdv = 1. / math.sqrt(Dk.size(1))
            Dk.data.uniform_(-stdv, stdv)
        for b in self.D_bias1:
            b.data.fill_(0)
        for b in self.D_bias2:
            b.data.fill_(0)
        for coe in self.ensemble_coef1:
            stdv = 1. / math.sqrt(coe.size(1))
            coe.data.uniform_(-stdv, stdv)
        for coe in self.ensemble_coef2:
            stdv = 1. / math.sqrt(coe.size(1))
            coe.data.uniform_(-stdv, stdv)
        for b in self.ensemble_bias1:
            b.data.fill_(0)
        for b in self.ensemble_bias2:
            b.data.fill_(0)

    def get_knn_graph(self,features,k=20):
        if not os.path.exists('saved_knn/'):
           os.mkdir('saved_knn')
        if not os.path.exists('saved_knn/knn_graph_{}.npz'.format(features.shape)):
            features[features!=0] = 1
            sims = cosine_similarity(features)
            np.save('saved_knn/cosine_sims_{}.npy'.format(features.shape), sims)

            sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
            for i in range(len(sims)):
                indices_argsort = np.argsort(sims[i])
                sims[i, indices_argsort[: -k]] = 0

            adj_knn = sp.csr_matrix(sims)
            sp.save_npz('saved_knn/knn_graph_{}.npz'.format(features.shape), adj_knn)
        else:
            print('loading saved_knn/knn_graph_{}.npz...'.format(features.shape))
            adj_knn = sp.load_npz('saved_knn/knn_graph_{}.npz'.format(features.shape))
        return preprocess_adj_noloop(adj_knn, self.device)

    def get_knn_graph_structures(self,adj, k=20):
        if not os.path.exists('saved_knn/'):
            os.mkdir('saved_knn')
        if not os.path.exists('saved_knn/knn_graph_structural{}.npz'.format(adj.shape,k)):
            G = from_numpy_matrix(adj)
            sim_rank_G = simrank_similarity(G)
            sims = []
            for key in sim_rank_G.keys():
                sims.append(list(sim_rank_G[key].values()))
            sims = np.array(sims)
            np.save('saved_knn/structural_sims_{}.npy'.format(adj.shape,k), sims)
            sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
            for i in range(len(sims)):
                indices_argsort = np.argsort(sims[i])
                sims[i, indices_argsort[: -k]] = 0
            adj_knn = sp.csr_matrix(sims)
            sp.save_npz('saved_knn/knn_graph_structural{}.npz'.format(adj.shape,k), adj_knn)
        else:
            print('loading saved_knn/knn_graph_structural{}.npz...'.format(adj.shape,k))
            adj_knn = sp.load_npz('saved_knn/knn_graph_structural{}.npz'.format(adj.shape,k))
        return preprocess_adj_noloop(adj_knn, self.device)
    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def fit(self, features, adj, labels, idx_train, idx_val=None,train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """
        self.fit_(features, adj, labels, idx_train, idx_val, train_iters=train_iters, initialize=initialize, verbose=verbose)



    def fit_(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)
    def forward(self, fea, adj):
        x = self.myforward(fea,adj)
        return x

    def myforward(self, fea, adj):
        '''output embedding and log_softmax'''
        if self.adj_knn_feature is None:
            self.adj_knn_feature = self.get_knn_graph(fea.to_dense().cpu().numpy())
        if self.adj_knn_structure is None:
            self.adj_knn_structure = self.get_knn_graph_structures_test(adj.to_dense().cpu().numpy(),self.k)
        adj_knn_structure = self.adj_knn_structure
        adj_knn_features = self.adj_knn_feature
        gamma = self.gamma

        s_i1 = torch.sigmoid(fea @ self.scores1[0] + self.bias1[0])
        s_i2 = torch.sigmoid(fea @ self.scores2[0] + self.bias2[0])
        Dk_i1 = (fea @ self.D_k1[0] + self.D_bias1[0])
        Dk_i2 = (fea @ self.D_k2[0] + self.D_bias2[0])
        x1 = (s_i1 * self.gc11(fea, adj) + (1-s_i1) * self.gc11(fea, adj_knn_structure)) + (gamma) * Dk_i1 * self.gc11(fea, self.identity)
        x2 = (s_i2 * self.gc21(fea, adj) + (1-s_i2) * self.gc21(fea, adj_knn_features)) + (gamma) * Dk_i2 * self.gc21(fea, self.identity)
        adj_dense = adj.to_dense()
        adj_knn_structure_dense = adj_knn_structure.to_dense()
        adj_knn_features_dense = adj_knn_features.to_dense()
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.dropout(x2,self.dropout,training=self.training)
        # output, no relu and dropput here.
        s_o1 = torch.sigmoid(x1 @ self.scores1[-1] + self.bias1[-1])
        s_o2 = torch.sigmoid(x2 @ self.scores2[-1] + self.bias2[-1])

        Dk_o1 = (x1 @ self.D_k1[-1] + self.D_bias1[-1])
        Dk_o2 = (x2 @ self.D_k2[-1] + self.D_bias1[-1])
        coef1 = torch.sigmoid(x1 @ self.ensemble_coef1[-1] + self.ensemble_bias1[-1])
        coef2 = torch.sigmoid(x2 @ self.ensemble_coef2[-1] + self.ensemble_bias2[-1])
        x1 = (s_o1 * self.gc12(x1, adj) + (1-s_o1) * self.gc12(x1, adj_knn_structure)) + (gamma) * Dk_o1 * self.gc12(x1, self.identity)
        x2 = (s_o2 * self.gc22(x2, adj) + (1-s_o2) * self.gc22(x2, adj_knn_features)) + (gamma) * Dk_o2 * self.gc22(x2, self.identity)
        x = coef1 * x1 + coef2 * x2
        x = F.log_softmax(x, dim=1)
        return x

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.myforward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            # def eval_class(output, labels):
            #     preds = output.max(1)[1].type_as(labels)
            #     return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro') + \
            #         f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # perf_sum = eval_class(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)



    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCNJaccard
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            adj = self.drop_dissimilar_edges(features, adj)
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)

def noaug_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj_noloop(adj, device):
    adj_normalizer = noaug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = utils.sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj

def __dropedge_jaccard(A, iA, jA, features, threshold):
    # deprecated: for sparse feature matrix...
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]

            intersection = a.multiply(b).count_nonzero()
            J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)

            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt

@njit
def dropedge_jaccard(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            intersection = np.count_nonzero(a*b)
            J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)

            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


@njit
def dropedge_cosine(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-8)

            if C < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt

@njit
def dropedge_dis(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C = np.linalg.norm(features[n1] - features[n2])
            if C > threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt

@njit
def dropedge_both(A, iA, jA, features, threshold1=2.5, threshold2=0.01):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C1 = np.linalg.norm(features[n1] - features[n2])

            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C2 = inner_product / (np.sqrt(np.square(a).sum() + np.square(b).sum())+ 1e-6)
            if C1 > threshold1 or threshold2 < 0:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt