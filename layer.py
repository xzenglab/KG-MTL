'''
@Author: tengfei ma
@Date: 2020-05-09 21:27:12
LastEditTime: 2021-05-29 07:00:00
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
    def forward(self,drug_cnn,drug_kg, variant='KG-MTL-C'):
        ##### linear
        if variant=='KG-MTL':
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
        elif variant=='KG-MTL-L':
            drug_cnn_=self.w_aa_.squeeze()*drug_cnn+self.w_ab_.squeeze()*drug_kg
            drug_kg_=self.w_ba_.squeeze()*drug_cnn+self.w_bb_.squeeze()*drug_kg
            return drug_cnn_, drug_kg_
        elif variant=='KG-MTL-C':
            drug_cnn=drug_cnn.unsqueeze(2)
            drug_kg=drug_kg.unsqueeze(1)
            
            c_mat=th.matmul(drug_cnn,drug_kg)
            c_mat_t=c_mat.permute(0, 2, 1)
    
            c_mat=c_mat.view(-1,self.out_dim)
            c_mat_t=c_mat.view(-1,self.out_dim)
            drug_cnn=(c_mat.matmul(self.w_aa)+c_mat_t.matmul(self.w_ab)).view(-1,self.out_dim)+self.d_cnn_bias.squeeze()
            drug_kg=(c_mat.matmul(self.w_ba)+c_mat_t.matmul(self.w_bb)).view(-1,self.out_dim)+self.d_kg_bias.squeeze()
            return drug_cnn, drug_kg

class AttentionUnit(nn.Module):
    def __init__(self, input_dim=128):
        super(AttentionUnit, self).__init__()
        self.dim=input_dim
        ### shared parameters
        self.W = nn.Parameter(th.Tensor(input_dim,1))
        nn.init.xavier_uniform_(self.W)
        
        self.W_cpi = nn.Parameter(th.Tensor(input_dim,1))
        nn.init.xavier_uniform_(self.W_cpi)
        self.W_dti = nn.Parameter(th.Tensor(input_dim,1))
        nn.init.xavier_uniform_(self.W_dti)
        self.W_a = nn.Parameter(th.Tensor(input_dim,2))
        nn.init.xavier_uniform_(self.W_a)
        #self.MLP_dti=nn.Linear(2*input_dim, input_dim)
        self.MLP_cpi=nn.Linear(2*input_dim, input_dim)
    def forward(self, drug_cnn, drug_kg):
        drug_cnn_=drug_cnn.unsqueeze(1)
        drug_kg_=drug_kg.unsqueeze(1)
        features=drug_cnn_+drug_kg_
        features=features.squeeze(1)
        features=F.softmax(th.matmul(th.tanh(features), self.W_a))
        features=features.unsqueeze(1)
        drug_cnn= (features[:,:,0].unsqueeze(1)* drug_cnn_).squeeze()
        
        drug_kg=(features[:,:,1].unsqueeze(1)*drug_kg_).squeeze()
        features=th.cat((drug_cnn, drug_kg), dim=1)
        features_cpi=self.MLP_cpi(features)
        #features_dti=self.MLP_cpi(features)
        return features_cpi, features_cpi


if __name__=='__main__':
    au=AttentionUnit(input_dim=10)
    drug_cnn=th.Tensor(1,10)
    drug_kg=th.Tensor(1,10)
    au(drug_cnn, drug_kg)
        