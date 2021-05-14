'''
@Author: tengfei ma
@Date: 2020-05-09 21:27:12
LastEditTime: 2021-05-12 07:21:23
LastEditors: Please set LastEditors
@Description: RGCN与共享
@FilePath: /Multi-task-pytorch/layer.py
'''
import dgl
import dgl.function as fn
import torch as th
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import RelGraphConv
import numpy as np
import torch.nn as nn


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    '''
    @description: 
    @param {type} 构建rgcn模型
    @return: 
    '''

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h


class EmbeddingLayer(nn.Module):
    '''
    @description: 
    @param {type} num_nodes: kg中结点的数量，h_dim: 要使用的隐层数目
    @return: 
    '''

    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = th.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers-1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, 'bdd', self.num_bases, activation=act, self_loop=True, dropout=self.dropout)

#Cross-stitch https://arxiv.org/abs/1604.03539
class Cross_stitch(nn.Module):
    def __init__(self):
        super(Cross_stitch,self).__init__()
        # self.out_dim=out_dim
        self.w_aa = nn.Parameter(th.Tensor(1,))
        self.w_aa.data=th.tensor(np.random.random(),requires_grad=True)
        
        self.w_ab=nn.Parameter(th.Tensor(1,))
        self.w_ab.data=th.tensor(np.random.random(),requires_grad=True)
        self.w_ba=nn.Parameter(th.Tensor(1,))
        self.w_ba.data=th.tensor(np.random.random(),requires_grad=True)
        self.w_bb=nn.Parameter(th.Tensor(1,))
        self.w_bb.data=th.tensor(np.random.random(),requires_grad=True)
        # np.random.random()
        
        
        print(self.w_aa)
    def forward(self,drug_cnn,drug_kg):
        drug_cnn_=self.w_aa*drug_cnn+self.w_ab*drug_kg
        drug_kg_=self.w_ba*drug_cnn+self.w_bb*drug_kg
        
        print('shared parameters: w_aa:{:.4f}, w_ab:{:.4f}, w_ba:{:.4f}, w_bb:{:.4f}'.format(self.w_aa,self.w_ab,self.w_ba,self.w_bb))
        
        return drug_cnn_,drug_kg_

#非线性的参数共享
class Shared_Unit_NL(nn.Module):
    def __init__(self,input_dim=200):
        super(Shared_Unit_NL,self).__init__()
        self.out_dim=input_dim
        self.w_aa = nn.Parameter(th.Tensor(input_dim,1))
        nn.init.xavier_uniform_(self.w_aa)
        
        self.w_ab=nn.Parameter(th.Tensor(input_dim,1))
        nn.init.xavier_normal_(self.w_ab)
        
        self.w_ba=nn.Parameter(th.Tensor(input_dim,1))
        nn.init.xavier_normal_(self.w_ba)
        self.w_bb=nn.Parameter(th.Tensor(input_dim,1))
        nn.init.xavier_normal_(self.w_bb)

        self.d_cnn_bias=nn.Parameter(th.Tensor(input_dim,1))
        self.d_kg_bias=nn.Parameter(th.Tensor(input_dim,1))

        nn.init.xavier_normal_(self.d_cnn_bias)
        nn.init.xavier_normal_(self.d_kg_bias)
        
        self.w_aa_ = nn.Parameter(th.Tensor(input_dim,1))
        nn.init.xavier_uniform_(self.w_aa_)
        
        self.w_ab_=nn.Parameter(th.Tensor(input_dim,1))
        nn.init.xavier_uniform_(self.w_ab_)
        
        self.w_ba_=nn.Parameter(th.Tensor(input_dim,1))
        nn.init.xavier_uniform_(self.w_ba_)
        
        self.w_bb_=nn.Parameter(th.Tensor(input_dim,1))
        nn.init.xavier_uniform_(self.w_bb_)
       
        
        #print(self.w_aa)
    def forward(self,drug_cnn,drug_kg):
        ##### linear
        drug_cnn_=self.w_aa_.squeeze()*drug_cnn+self.w_ab_.squeeze()*drug_kg
        drug_kg_=self.w_ba_.squeeze()*drug_cnn+self.w_bb_.squeeze()*drug_kg
        
        # #### non-linear
        drug_cnn=drug_cnn_.unsqueeze(2)
        drug_kg=drug_kg_.unsqueeze(1)
        
        c_mat=th.matmul(drug_cnn,drug_kg)
        c_mat_t=c_mat.permute(0, 2, 1)

        c_mat=c_mat.view(-1,self.out_dim)
        c_mat_t=c_mat.view(-1,self.out_dim)
        drug_cnn=(c_mat.matmul(self.w_aa)+c_mat_t.matmul(self.w_ab)).view(-1,self.out_dim)+self.d_cnn_bias.squeeze()
        drug_kg=(c_mat.matmul(self.w_ba)+c_mat_t.matmul(self.w_bb)).view(-1,self.out_dim)+self.d_kg_bias.squeeze()
        
        
        return drug_cnn, drug_kg

        