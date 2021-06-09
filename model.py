
'''
@Author: your name
@Date: 2020-05-15 10:12:31
LastEditTime: 2021-05-29 12:57:15
LastEditors: Please set LastEditors
@Description: DTi二分类与DDI多任务结合
@FilePath: /Multi-task-pytorch/model.py
'''
from layer import *
import utils
import argparse
import numpy as np
import time
import random
import torch
from torch.autograd import Variable
from dgl import DGLGraph
from dgl.nn.pytorch import RelGraphConv
import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv,MaxPooling,GraphConv
from dgllife.model.readout.mlp_readout import MLPNodeReadout
class EmbeddingLayer(nn.Module):
    '''
    @description:
    @param {type} num_nodes: kg中结点的数量，h_dim: 要使用的隐层数目
    @return:
    '''

    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, h):
        return self.embedding(h.squeeze())


class MKDTI(nn.Module):
    def __init__(self,shared_unit_num, drug_hidden_dim, protein_size, cpi_fc_layers, dti_fc_layers, dropout_prob, num_nodes, h_dim, out_dim, num_rels, num_bases, num_hidden_layers=2, dropout=0.5, use_self_loop=False, use_cuda=False, reg_param=0, device='cpu',variant='KG-MTL'):
        super(MKDTI, self).__init__()
        # self.rgcn = RGCN(num_nodes, h_dim, out_dim, num_rels, num_bases,
        #                  num_hidden_layers, dropout, use_self_loop, use_cuda)
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.device=device
        self.num_hidden_layers = num_hidden_layers
        self.entity_embedding = EmbeddingLayer(num_nodes, h_dim)
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(
            self.w_relation, gain=nn.init.calculate_gain('relu'))

        self.rgcn_layers = nn.ModuleList([self.entity_embedding])
        self._construct_rgcn(num_hidden_layers)
        self.reg_param = reg_param
        self.shared_units=nn.ModuleList()
        for i in range(shared_unit_num):
            self.shared_units.append(Shared_Unit_NL(h_dim,variant=variant))
            #self.shared_units.append(SimpleUnit(h_dim))
            #self.shared_units.append(AttentionUnit(h_dim))
            #self.shared_units.append(Cross_stitch())
        #self.drug_size = drug_size
        self.drug_hidden_dim=drug_hidden_dim
        self.protein_size = protein_size

        #self.layer_filters_drugs = [50, 1, 1, 32,64,128,200]
        self.layer_filters_proteins = [self.drug_hidden_dim, 96, 128, self.drug_hidden_dim]
        self.cpi_hidden_dim = [78,self.drug_hidden_dim, self.drug_hidden_dim]
        # 设置共享层数-做消融实验
        self.num_shared_layer=shared_unit_num #1, 2, 3 layers

        self.compound_hidden_dim=[self.drug_hidden_dim,self.drug_hidden_dim,self.drug_hidden_dim]
        self.dti_hidden_dim = [2*self.drug_hidden_dim, 2*self.drug_hidden_dim, 2*self.drug_hidden_dim]
        self.kernals = [5, 7, 9,11]
        # SMILES encoding
        # self.embed_drug = nn.Embedding(
        #     num_embeddings=drug_size, embedding_dim=50)
        # self.drug_cnn = nn.ModuleList([nn.Conv1d(in_channels=self.layer_filters_drugs[i], out_channels=self.layer_filters_drugs[i+1],
        #                                          kernel_size=self.kernals[i], padding=int(self.kernals[i]/2)) for i in range(len(self.layer_filters_drugs)-1)])
        
        self.drug_gcn=nn.ModuleList([GraphConv(in_feats=self.cpi_hidden_dim[i],out_feats=self.cpi_hidden_dim[i+1]) for i in range(len(self.cpi_hidden_dim)-1)])
        # 对分子图做池化操作
        self.drug_output_layer=MLPNodeReadout(self.drug_hidden_dim,self.drug_hidden_dim,self.drug_hidden_dim, activation=nn.ReLU(),mode='max')
        self.compound_fc_layers=nn.ModuleList([nn.Linear(self.compound_hidden_dim[i],self.compound_hidden_dim[i+1]) for i in range(len(self.compound_hidden_dim)-1)])
        # protein encoding
        self.embed_protein = nn.Embedding(
            num_embeddings=protein_size, embedding_dim=self.drug_hidden_dim)

        self.target_cnn = nn.ModuleList([nn.Conv1d(in_channels=self.layer_filters_proteins[i], out_channels=self.layer_filters_proteins[i+1],
                                                   kernel_size=3, padding=1) for i in range(len(self.layer_filters_proteins)-1)])

        self.fc_layers = nn.ModuleList()
        for i in range(cpi_fc_layers):
            if i == cpi_fc_layers-1:
                # self.fc_layers.append(nn.Linear(self.h_dim, 2))
                # self.fc_layers.append(nn.Softmax(dim=1))
                self.fc_layers.append(nn.Linear(self.dti_hidden_dim[i], 1))
                #self.fc_layers.append(nn.Sigmoid())
            else:
                self.fc_layers.append(
                    nn.Linear(self.dti_hidden_dim[i], self.dti_hidden_dim[i+1]))
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(dropout_prob))

        self.dti_fc_layers = nn.ModuleList()
        for i in range(dti_fc_layers):
            if i == dti_fc_layers-1:
                # self.dti_fc_layers.append(nn.Linear(2*self.h_dim, 2))
                # self.dti_fc_layers.append(nn.Softmax(dim=1))
                self.dti_fc_layers.append(nn.Linear(self.dti_hidden_dim[i], 1))
                #self.dti_fc_layers.append(nn.Sigmoid())
            else:
                self.dti_fc_layers.append(
                    nn.Linear(self.dti_hidden_dim[i], self.dti_hidden_dim[i+1]))
                self.dti_fc_layers.append(nn.ReLU())
                self.dti_fc_layers.append(nn.Dropout(dropout_prob))

    '''
    @description: 获取CNN模块处理之后的数据维度
    @param {type}
    @return:
    '''

    def _construct_rgcn(self, hidden_rgcn_layers):
        for idx in range(hidden_rgcn_layers):
            act = F.relu if idx < self.num_hidden_layers-1 else None
            self.rgcn_layers.append(RelGraphConv(self.h_dim, self.h_dim, self.num_rels,
                                                 'bdd', num_bases=self.num_bases, activation=act, self_loop=True, dropout=self.dropout))

    def MT_Net(self, protein_seq, compound_indexs, drugs_entityid, target_entityid,compound_graphs,compound_vector, g, h, r, norm):
        # smiles encoding
        # compound_vector = self.embed_drug(compound_smiles)
        # compound_vector = compound_vector.permute(0, 2, 1)
        #compound_graphs=dgl.add_self_loop(compound_graphs)
        for l in self.drug_gcn:
            compound_vector=F.relu(l(compound_graphs,compound_vector))
        
        compound_vector=F.relu(self.drug_output_layer(compound_graphs,compound_vector))
        h = self.rgcn_layers[0](h)
        for idx, l in enumerate(self.compound_fc_layers):
            # 对输入特征做处理
            # 对输入特征做处理
            compound_vector = F.relu(l(compound_vector))
            if idx <= self.num_hidden_layers-1:
                h = self.rgcn_layers[idx+1](g, h, r, norm)
                ## KG-MTL-S
                if idx<=(self.num_shared_layer-1):
                    compound_embed = h[compound_indexs, :]
                    compound_vector, compound_kg_ = self.shared_units[idx](
                        compound_vector.squeeze(), compound_embed)
                    h[compound_indexs,:]=compound_kg_.clone()
        # protein encoding
        protein_vector = self.embed_protein(protein_seq)
        protein_vector = protein_vector.permute(0, 2, 1)
        for l in self.target_cnn:
            protein_vector = F.relu(l(protein_vector))
        protein_vector = F.adaptive_max_pool1d(protein_vector, output_size=1)
        protein_vector = protein_vector.view(protein_vector.size(0), -1)
        #.squeeze()
        output_cpi_vector = torch.cat(
            (compound_vector, protein_vector), dim=1)
        for fc_layer in self.fc_layers:
            output_cpi_vector = fc_layer(output_cpi_vector)
        drugs_embed = h[drugs_entityid, :]
        targets_embed = h[target_entityid, :]
        output_dti_vector = torch.cat((drugs_embed, targets_embed), dim=1)
        for fc_layer in self.dti_fc_layers:
            output_dti_vector = fc_layer(output_dti_vector)

        return output_cpi_vector, output_dti_vector, h

    def RGCN_Net(self, g, h, r, norm):
        for idx, layer in enumerate(self.rgcn_layers):
            if idx == 0:
                h = layer(h)
                continue
            h = layer(g, h, r, norm)
        return h

    def get_graphs_features(self,compound_smiles,smiles2graph):
        h=list()
        graphs=list()
        for c_id in compound_smiles:
            c_size, features, edge_index=smiles2graph[c_id]
            g=DGLGraph()
            #g=dgl.graph()
            g.add_nodes(c_size)
            if edge_index:
                edge_index=np.array(edge_index)
                g.add_edges(edge_index[:,0],edge_index[:,1])
            
            for f in features:
                h.append(f)
            g.ndata['x']=torch.from_numpy(np.array(features))
            g=dgl.add_self_loop(g)
            # g=g.to(torch.device('cuda:0'))
            # print(g.device)
            graphs.append(g)
        g=dgl.batch(graphs)
        #g=g.to(torch.device('cuda:0'))
        return g,torch.from_numpy(np.array(h))

    def cal_score(self, embedding, triples):
        s = embedding[triples[:, 0]]
        r = self.w_relation[triples[:, 1]]
        o = embedding[triples[:, 2]]
        score = th.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm, compound_smiles=None, protein_seq=None, compound_entityid=None, drugs_entityid=None, targets_entityid=None, rgcn_only=False,smiles2graph=None,eval_=False):
        compound_graphs,compound_features=self.get_graphs_features(compound_smiles,smiles2graph)
        if eval_:
            compound_features=compound_features.float()
            
        else:
            compound_features=compound_features.float().cuda()
            compound_graphs=compound_graphs.to(torch.device(self.device))
            g=g.to(torch.device(self.device))
        if rgcn_only:
            embed = self.RGCN_Net(g, h, r, norm)
            return embed
        else:
            node_map = dict()
            for i in range(len(h)):
                node_map[int(h[i])] = i
            indexs = list()
            for i in list(compound_entityid):
                indexs.append(node_map[i])

            drugs_index = list()
            targets_index = list()
            for i in drugs_entityid:
                drugs_index.append(node_map[i])
            for i in targets_entityid:
                targets_index.append(node_map[i])
            return self.MT_Net(protein_seq, np.array(indexs), np.array(drugs_index), np.array(targets_index),compound_graphs,compound_features, g, h, r, norm)

    def regularization_loss(self, embed):
        return th.mean(embed.pow(2))+th.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triples, labels):
        score = self.cal_score(embed, triples)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss+reg_loss

# cpi任务的独立模块


class CPI(nn.Module):
    def __init__(self, drug_size, protein_size, cpi_fc_layers, dropout_prob, h_dim=200):
        super(CPI, self).__init__()
        self.drug_size = drug_size
        self.protein_size = protein_size
        self.h_dim = h_dim
        self.layer_filters_drugs = [50, 1, 1, 32,64,128,200]
        self.layer_filters_proteins = [50, 96, 128, 200]
        self.cpi_hidden_dim = [400, 200, 100]
        self.kernals = [5, 7, 9,11,11,11]
        # SMILES encoding
        self.embed_drug = nn.Embedding(
            num_embeddings=drug_size, embedding_dim=50)
        self.drug_cnn = nn.ModuleList([nn.Conv1d(in_channels=self.layer_filters_drugs[i], out_channels=self.layer_filters_drugs[i+1],
                                                 kernel_size=self.kernals[i], padding=int(self.kernals[i]/2)) for i in range(len(self.layer_filters_drugs)-1)])
        # protein encoding
        self.embed_protein = nn.Embedding(
            num_embeddings=protein_size, embedding_dim=50)
        
        self.target_cnn = nn.ModuleList([nn.Conv1d(in_channels=self.layer_filters_proteins[i], out_channels=self.layer_filters_proteins[i+1],
                                                   kernel_size=self.kernals[i], padding=int(self.kernals[i]/2)) for i in range(len(self.layer_filters_proteins)-1)])
        # self.rnn_drug=nn.LSTM(input_size = self.layer_filters_drugs[-1],hidden_size=drug_size,num_layers=2,bidirectional=True,batch_first = True)
        # self.rnn_protein=nn.LSTM(input_size = self.layer_filters_proteins[-1],hidden_size=protein_size,num_layers=2,bidirectional=True,batch_first = True)
        self.fc_layers = nn.ModuleList()
        # self.extract_features_drug=nn.Linear(80000,self.h_dim)
        # self.extract_features_protein=nn.Linear(80000,self.h_dim)
        for i in range(cpi_fc_layers):
            if i == cpi_fc_layers-1:
                self.fc_layers.append(nn.Linear(self.cpi_hidden_dim[i], 1))
                self.fc_layers.append(nn.Sigmoid())
            else:
                self.fc_layers.append(
                    nn.Linear(self.cpi_hidden_dim[i], self.cpi_hidden_dim[i+1]))
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(dropout_prob))
    # CPI任务单独使用的模型

    def CPI_Net(self, compound_smiles, protein_seq):
        compound_vector = self.embed_drug(compound_smiles)
        compound_vector = compound_vector.permute(0, 2, 1)
        for idx, l in enumerate(self.drug_cnn):
            compound_vector = F.relu(l(compound_vector))
        compound_vector = F.adaptive_max_pool1d(
            compound_vector, output_size=1)
        compound_vector = compound_vector.view(compound_vector.size(0), -1)
        # batch_size = compound_vector.size(0)
        # compound_vector = compound_vector.view(compound_vector.size(0), compound_vector.size(2), -1)
        # direction = 2

        # h0 = torch.randn(2 * direction, batch_size, self.drug_size).cuda()
        # c0 = torch.randn(2 * direction, batch_size, self.drug_size).cuda()
        # compound_vector, (hn, cn) = self.rnn_drug(compound_vector, (h0, c0))
        # compound_vector=torch.flatten(compound_vector,1)
        # compound_vector=self.extract_features_drug(compound_vector)
        # protein encoding
        protein_vector = self.embed_protein(protein_seq)
        protein_vector = protein_vector.permute(0, 2, 1)
        for l in self.target_cnn:
            protein_vector = F.relu(l(protein_vector))

        protein_vector = F.adaptive_max_pool1d(protein_vector, output_size=1)
        protein_vector = protein_vector.view(protein_vector.size(0), -1)

        output_cpi_vector = torch.cat(
            (compound_vector, protein_vector.squeeze()), dim=1)
        for fc_layer in self.fc_layers:
            output_cpi_vector = fc_layer(output_cpi_vector)

        return output_cpi_vector

    def forward(self, compound_smiles, protein_seq):
        return self.CPI_Net(compound_smiles, protein_seq)


class DTI(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases, num_hidden_layers=2, dropout=0, use_self_loop=False, use_cuda=False, reg_param=0):
        super(DTI, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.dti_hidden_dim = [2*h_dim, 2*h_dim]
        self.entity_embedding = EmbeddingLayer(num_nodes, h_dim)
        self.rgcn_layers = nn.ModuleList([self.entity_embedding])
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(
            self.w_relation,gain=nn.init.calculate_gain('relu'))
        ###, 
        self.num_hidden_layers = num_hidden_layers
        self._construct_rgcn(num_hidden_layers)
        
        self.dti_fc_layers = nn.ModuleList()
        for i in range(len(self.dti_hidden_dim)):
            if i == len(self.dti_hidden_dim)-1:
                self.dti_fc_layers.append(nn.Linear(self.dti_hidden_dim[i], 1))
                self.dti_fc_layers.append(nn.Sigmoid())
            else:
                self.dti_fc_layers.append(
                    nn.Linear(self.dti_hidden_dim[i], self.dti_hidden_dim[i+1]))
                self.dti_fc_layers.append(nn.ReLU())
                self.dti_fc_layers.append(nn.Dropout(dropout))

    def _construct_rgcn(self, hidden_rgcn_layers):
        for idx in range(hidden_rgcn_layers):
            act = F.relu if idx < self.num_hidden_layers-1 else None
            self.rgcn_layers.append(RelGraphConv(self.h_dim, self.h_dim, self.num_rels,
                                                 'bdd', num_bases=self.num_bases, activation=None, self_loop=True, dropout=self.dropout))

    def RGCN_Net(self, g, h, r, norm):
        h=self.rgcn_layers[0](h)
        for idx in range(1, len(self.rgcn_layers)):
            layer=self.rgcn_layers[idx]
            h = layer(g, h, r, norm)
        return h

    def Net(self, drugs_entityid, target_entityid, g, h, r, norm):
        h = self.RGCN_Net(g, h, r, norm)
        drugs_embed = h[drugs_entityid, :]
        targets_embed = h[target_entityid, :]
        output_dti_vector = torch.cat((drugs_embed, targets_embed), dim=1)
        for fc_layer in self.dti_fc_layers:
            output_dti_vector = fc_layer(output_dti_vector)

        return output_dti_vector, h

    def forward(self, drugs_entityid, targets_entityid, g, h, r, norm):

        node_map = dict()
        for i in range(len(h)):
            node_map[int(h[i])] = i

        drugs_index = list()
        targets_index = list()
        for i in drugs_entityid:
            drugs_index.append(node_map[i])
        for i in targets_entityid:
            targets_index.append(node_map[i])
        return self.Net(np.array(drugs_index), np.array(targets_index), g, h, r, norm)


class CPI_MPNN(nn.Module):
    def __init__(self, mpnn_hidden_size, mpnn_depth, drug_size, protein_size, cpi_fc_layers, dropout_prob, h_dim=200):
        super(CPI_MPNN, self).__init__()
        self.mpnn_hidden_size = mpnn_hidden_size
        self.mpnn_depth = mpnn_depth
        self.drug_size = drug_size
        self.protein_size = protein_size
        self.cpi_fc_layers = cpi_fc_layers
        self.layer_filters_drugs = [50, 1, 1, 1]
        self.layer_filters_proteins = [50, 96, 128, 200]
        self.cpi_hidden_dim = [400, 200, 100]
        self.embed_protein = nn.Embedding(
            num_embeddings=protein_size, embedding_dim=50)
        self.target_cnn = nn.ModuleList([nn.Conv1d(in_channels=self.layer_filters_proteins[i], out_channels=self.layer_filters_proteins[i+1],
                                                   kernel_size=self.kernals[i], padding=int(self.kernals[i]/2)) for i in range(len(self.layer_filters_proteins)-1)])
        from chemutils import ATOM_FDIM, BOND_FDIM

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM,
                             self.mpnn_hidden_size, bias=False)
        self.W_h = nn.Linear(self.mpnn_hidden_size,
                             self.mpnn_hidden_size, bias=False)
        self.W_o = nn.Linear(
            ATOM_FDIM + self.mpnn_hidden_size, self.mpnn_hidden_size)

    def single_molecule_forward(self, fatoms, fbonds, agraph, bgraph):
        '''
                fatoms: (x, 39)
                fbonds: (y, 50)
                agraph: (x, 6)
                bgraph: (y,6)
        '''
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        agraph = create_var(agraph)
        bgraph = create_var(bgraph)
        binput = self.W_i(fbonds)
        message = F.relu(binput)
        #print("shapes", fbonds.shape, binput.shape, message.shape)
        for i in range(self.mpnn_depth - 1):
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            message = F.relu(binput + nei_message)
        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = F.relu(self.W_o(ainput))
        return torch.mean(atom_hiddens, 0).view(1, -1).to(device)

    def forward(self, feature, protein_seq):
        '''
                        batch_size == 1
                        feature: utils.smiles2mpnnfeature
                '''
        fatoms, fbonds, agraph, bgraph, atoms_bonds = feature
        agraph = agraph.long()
        bgraph = bgraph.long()
        #print(fatoms.shape, fbonds.shape, agraph.shape, bgraph.shape, atoms_bonds.shape)
        atoms_bonds = atoms_bonds.long()
        batch_size = atoms_bonds.shape[0]
        N_atoms, N_bonds = 0, 0
        embeddings = []
        for i in range(batch_size):
            n_a = atoms_bonds[i, 0].item()
            n_b = atoms_bonds[i, 1].item()
            if (n_a == 0):
                embed = create_var(torch.zeros(1, self.mpnn_hidden_size))
                embeddings.append(embed.to(device))
                continue
            sub_fatoms = fatoms[N_atoms:N_atoms+n_a, :].to(device)
            sub_fbonds = fbonds[N_bonds:N_bonds+n_b, :].to(device)
            sub_agraph = agraph[N_atoms:N_atoms+n_a, :].to(device)
            sub_bgraph = bgraph[N_bonds:N_bonds+n_b, :].to(device)
            embed = self.single_molecule_forward(
                sub_fatoms, sub_fbonds, sub_agraph, sub_bgraph)
            embed = embed.to(device)
            embeddings.append(embed)
            N_atoms += n_a
            N_bonds += n_b
        try:
            embeddings = torch.cat(embeddings, 0)
        except:
            #embeddings = torch.cat(embeddings, 0)
            print(embeddings)
        
        protein_vector = self.embed_protein(protein_seq)
        protein_vector = protein_vector.permute(0, 2, 1)
        for l in self.target_cnn:
            protein_vector = F.relu(l(protein_vector))

        protein_vector = F.adaptive_max_pool1d(protein_vector, output_size=1)
        protein_vector = protein_vector.view(protein_vector.size(0), -1)

        output_cpi_vector = torch.cat(
            (embeddings, protein_vector.squeeze()), dim=1)
        for fc_layer in self.fc_layers:
            output_cpi_vector = fc_layer(output_cpi_vector)
        
        return output_cpi_vector

class MultiTaskLoss(nn.Module):
    def __init__(self,task_num,shared_unit_num,drug_hidden_dim, protein_size, cpi_fc_layers, dti_fc_layers, dropout_prob, num_nodes, h_dim, out_dim, num_rels, num_bases, num_hidden_layers=3, dropout=0.5, use_self_loop=False, use_cuda=False, reg_param=0, variant='KG-MTL', device='cpu'):
        super(MultiTaskLoss,self).__init__()
        self.task_num=task_num
        self.log_var=nn.Parameter(torch.zeros(task_num, requires_grad=True))
        self.log_var
        #nn.init.xavier_uniform_(self.log_var)
        self.multi_task=MKDTI(shared_unit_num,drug_hidden_dim, protein_size, cpi_fc_layers, dti_fc_layers, dropout_prob, num_nodes, h_dim, out_dim, num_rels, num_bases, variant=variant, device=device)
    def _calc_loss(self, cpi_loss, dti_loss, mode='weighted'):
        if mode=='weighted':
            pre1=torch.exp(-self.log_var[0])    
            pre2=torch.exp(-self.log_var[1])
            loss=torch.sum(pre1 * cpi_loss + self.log_var[0], -1)
            loss+=torch.sum(pre2 * dti_loss + self.log_var[1], -1)
            loss=torch.mean(loss)
            return loss
        elif mode=='single':
            #loss=cpi_loss+dti_loss
            loss=torch.sum(torch.cat(cpi_loss, dti_loss))
            return loss
    
    def forward(self,g, h, r, norm, compound_smiles=None, protein_seq=None, compound_entityid=None, drugs_entityid=None, targets_entityid=None, rgcn_only=False,smiles2graph=None,eval_=False,cpi_labels=None,dti_labels=None, mode='weighted'):
        if eval_:

            cpi_pred, dti_pred, _=self.multi_task(g,h,r,norm,compound_smiles,protein_seq,compound_entityid,drugs_entityid,targets_entityid,rgcn_only,smiles2graph,eval_)
            return F.sigmoid(cpi_pred),F.sigmoid(dti_pred)
        else:
            cpi_pred, dti_pred, _=self.multi_task(g,h,r,norm,compound_smiles,protein_seq,compound_entityid,drugs_entityid,targets_entityid,rgcn_only,smiles2graph,eval_)
            cpi_loss=F.binary_cross_entropy(F.sigmoid(cpi_pred), cpi_labels)
            dti_loss=F.binary_cross_entropy(F.sigmoid(dti_pred), dti_labels)
            loss = self._calc_loss(cpi_loss, dti_loss, mode=mode)
            return loss,cpi_loss,dti_loss, F.sigmoid(cpi_pred),F.sigmoid(dti_pred),self.log_var.data.tolist()

class CPI_GAT(nn.Module):
    def __init__(self,drug_size,protein_size, cpi_fc_layers, dropout_prob,num_layers,in_dim,num_hidden,heads,activate,feat_drop,attn_drop,negative_slope,residual):

        super(CPI_GCN,self).__init__()
        #self.g=g
        self.num_layers=num_layers
        self.gat_layers=nn.ModuleList()
        self.activate=activate
        self.drug_size=drug_size
        # input projection (no residual)
        self.gat_layers.append(GATConv(in_dim,num_hidden,heads[0],feat_drop,attn_drop,negative_slope,False,activate=self.activate))

        # hidden layers
        for i in range(1,num_layers):
            # due to multi-head, the in_dim=num_hidden * num_heads
            self.gat_layers.append(GATConv(num_hidden*heads[l-1],num_hidden,heads[i],feat_drop,attn_drop,negative_slope,residual,self.activate))

        # output projection
        self.gat_layers.append(GATConv(num_hidden*heads[-2],self.drug_size,heads[-1],feat_drop,attn_drop,negative_slope,residual,self.activate))
        
        self.embed_protein = nn.Embedding(
                    num_embeddings=protein_size, embedding_dim=50)

        self.target_cnn = nn.ModuleList([nn.Conv1d(in_channels=self.layer_filters_proteins[i], out_channels=self.layer_filters_proteins[i+1],
                                                   kernel_size=self.kernals[i], padding=int(self.kernals[i]/2)) for i in range(len(self.layer_filters_proteins)-1)])
    
    
    def get_graphs_features(self,compound_smiles,smiles2graph):
        h=list()
        graphs=list()
        for c_id in compound_smiles:
            c_size, features, edge_index=smiles2graph[c_id]
            g=DGLGraph()
            g.add_edges(edge_index)
            graphs.append(g)
            for f in features:
                h.append(f)
        g=dgl.batch(graphs)
        return g,h

    def forward(self,compound_smiles,protein_seq,smiles2graph):
        g,h=self.get_graphs_features(compound_smiles,smiles2graph)
        for l in self.gat_layers:
            pass


        # protein encoding
        protein_vector = self.embed_protein(protein_seq)
        protein_vector = protein_vector.permute(0, 2, 1)
        for l in self.target_cnn:
            protein_vector = F.relu(l(protein_vector))

        protein_vector = F.adaptive_max_pool1d(protein_vector, output_size=1)
        protein_vector = protein_vector.view(protein_vector.size(0), -1)

        output_cpi_vector = torch.cat(
            (compound_vector, protein_vector.squeeze()), dim=1)
        for fc_layer in self.fc_layers:
            output_cpi_vector = fc_layer(output_cpi_vector)

        return output_cpi_vector

class CPI_GCN(nn.Module):
    def __init__(self,in_dim,hidden_dim,drug_size,protein_size,dropout_prob=0.2,cpi_fc_layers=3):
        super(CPI_GCN,self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3=GraphConv(hidden_dim,hidden_dim)
        # self.gat1=GATConv(in_dim,hidden_dim,8)
        # self.gat2=GATConv(hidden_dim,hidden_dim,1)
        self.conv1=self.conv1.float()
        self.conv2=self.conv2.float()
        self.conv4=GraphConv(hidden_dim,hidden_dim)
        self.conv3.float()
        self.output_linear=nn.Linear(hidden_dim,drug_size)
        self.l_hidden_dim=[100,200]
        self.compound_fc_layers=nn.ModuleList()
        self.layer_filters_proteins = [200, 96, 128, 200]
        self.cpi_hidden_dim = [400,200,100,50]
        self.kernals = [3, 5, 7, 9, 9, 9]
        self.fc_layers=nn.ModuleList()
        self.embed_protein = nn.Embedding(
                    num_embeddings=protein_size, embedding_dim=200)

        self.target_cnn = nn.ModuleList([nn.Conv1d(in_channels=self.layer_filters_proteins[i], out_channels=self.layer_filters_proteins[i+1],
                                                   kernel_size=3, padding=1) for i in range(len(self.layer_filters_proteins)-1)])
        for i in range(len(self.l_hidden_dim)):
            if i == len(self.l_hidden_dim)-1:
                continue
            else:
                self.compound_fc_layers.append(
                    nn.Linear(self.l_hidden_dim[i], self.l_hidden_dim[i+1]))
                self.compound_fc_layers.append(nn.ReLU())
                self.compound_fc_layers.append(nn.Dropout(dropout_prob))
        
        for i in range(cpi_fc_layers):
            if i == cpi_fc_layers-1:
                self.fc_layers.append(nn.Linear(self.cpi_hidden_dim[i], 1))
                self.fc_layers.append(nn.Sigmoid())
            else:
                self.fc_layers.append(
                    nn.Linear(self.cpi_hidden_dim[i], self.cpi_hidden_dim[i+1]))
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(dropout_prob))
    def get_graphs_features(self,compound_smiles,smiles2graph):
        h=list()
        graphs=list()
        for c_id in compound_smiles:
            c_size, features, edge_index=smiles2graph[c_id]
            g=DGLGraph()
            g.add_nodes(c_size)
            if edge_index:
                edge_index=np.array(edge_index)
                g.add_edges(edge_index[:,0],edge_index[:,1])
            
            for f in features:
                h.append(f)
            g.ndata['x']=torch.from_numpy(np.array(features))
            graphs.append(g)
        g=dgl.batch(graphs)
        return g,torch.from_numpy(np.array(h))

    def forward(self,compound_smiles,protein_seq,smiles2graph,eval_=False):
        g,h=self.get_graphs_features(compound_smiles,smiles2graph)
        #h=h.double()
        if eval_:
            h=h.float()
        else:
            h=h.float().cuda()
        h=F.relu(self.conv1(g,h))
        h=F.relu(self.conv2(g,h))
        h=F.relu(self.conv3(g,h))
        h=F.relu(self.conv4(g,h))
        g.ndata['h']=h
        hg=dgl.mean_nodes(g,'h')
        #compound_vector=self.output_linear(hg)
        compound_vector=hg
        # for l in self.compound_fc_layers:
        #     compound_vector=l(compound_vector)
        #compound_vector=hg
        # protein encoding
        protein_vector = self.embed_protein(protein_seq)
        #protein_vector=protein_vector.unsqueeze(1)
        protein_vector = protein_vector.permute(0, 2, 1)
        for l in self.target_cnn:
            protein_vector = F.relu(l(protein_vector))
        protein_vector=protein_vector.squeeze()
        protein_vector=F.adaptive_max_pool1d(protein_vector,output_size=1)
        # protein_vector = F.adaptive_max_pool1d(protein_vector, output_size=1)
        
        #protein_vector = protein_vector.view(protein_vector.size(0), -1)

        output_cpi_vector = torch.cat(
            (compound_vector, protein_vector.squeeze()), dim=1)
        for fc_layer in self.fc_layers:
            output_cpi_vector = fc_layer(output_cpi_vector)

        return output_cpi_vector

class CPI_DGLLife(nn.Module):
    def __init__(self,in_dim,hidden_dim,drug_size,protein_size,dropout_prob=0.2,cpi_fc_layers=2,device='cpu'):
        super(CPI_DGLLife,self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3=GraphConv(hidden_dim,hidden_dim)
        self.device=device
        # self.gat1=GATConv(in_dim,hidden_dim,8)
        # self.gat2=GATConv(hidden_dim,hidden_dim,1)
        self.conv1=self.conv1.float()
        self.conv2=self.conv2.float()
        self.conv4=GraphConv(hidden_dim,hidden_dim)
        self.conv3.float()
        self.learn_graph1=MLPNodeReadout(hidden_dim,hidden_dim,hidden_dim)
        self.output_linear=nn.Linear(hidden_dim,drug_size)
        self.l_hidden_dim=[hidden_dim,hidden_dim,hidden_dim]
        self.compound_fc_layers=nn.ModuleList()
        self.layer_filters_proteins = [hidden_dim, 96, 128, in_dim,hidden_dim]
        # self.cpi_hidden_dim = [2*hidden_dim,2*hidden_dim,2*hidden_dim,2*hidden_dim]
        self.cpi_hidden_dim = [2*hidden_dim,2*hidden_dim,2*hidden_dim]
        self.kernals = [3, 5, 7, 9]
        self.fc_layers=nn.ModuleList()
        self.embed_protein = nn.Embedding(
                    num_embeddings=protein_size, embedding_dim=hidden_dim)

        self.target_cnn = nn.ModuleList([nn.Conv1d(in_channels=self.layer_filters_proteins[i], out_channels=self.layer_filters_proteins[i+1],
                                                   kernel_size=3, padding=1) for i in range(len(self.layer_filters_proteins)-1)])
        for i in range(len(self.l_hidden_dim)):
            if i == len(self.l_hidden_dim)-1:
                continue
            else:
                self.compound_fc_layers.append(
                    nn.Linear(self.l_hidden_dim[i], self.l_hidden_dim[i+1]))
                self.compound_fc_layers.append(nn.ReLU())
                self.compound_fc_layers.append(nn.Dropout(dropout_prob))
        
        for i in range(cpi_fc_layers):
            if i == cpi_fc_layers-1:
                self.fc_layers.append(nn.Linear(self.cpi_hidden_dim[i], 1))
                self.fc_layers.append(nn.Sigmoid())
            else:
                self.fc_layers.append(
                    nn.Linear(self.cpi_hidden_dim[i], self.cpi_hidden_dim[i+1]))
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(dropout_prob))
    def get_graphs_features(self,compound_smiles,smiles2graph):
        h=list()
        graphs=list()
        for c_id in compound_smiles:
            c_size, features, edge_index=smiles2graph[c_id]
            g=DGLGraph()
            g.add_nodes(c_size)
            if edge_index:
                edge_index=np.array(edge_index)
                g.add_edges(edge_index[:,0],edge_index[:,1])
            
            for f in features:
                h.append(f)
            g.ndata['x']=torch.from_numpy(np.array(features))
            g=dgl.add_self_loop(g)
            graphs.append(g)
        g=dgl.batch(graphs)
        return g,torch.from_numpy(np.array(h))

    def forward(self,compound_smiles,protein_seq,smiles2graph,eval_=False):
        g,h=self.get_graphs_features(compound_smiles,smiles2graph)
        #h=h.double()
        if eval_:
            h=h.float()
        else:
            h=h.float().cuda()
            g=g.to(torch.device(self.device))
        h=F.relu(self.conv1(g,h))
        # h=F.relu(self.conv2(g,h))
        # h=F.relu(self.conv3(g,h))
        hg=F.relu(self.learn_graph1(g,h))
        compound_vector=hg
        for l in self.compound_fc_layers:
            compound_vector=l(compound_vector)
        #compound_vector=hg
        # protein encoding
        protein_vector = self.embed_protein(protein_seq)
        #protein_vector=protein_vector.unsqueeze(1)
        protein_vector = protein_vector.permute(0, 2, 1)
        for l in self.target_cnn:
            protein_vector = F.relu(l(protein_vector))
        protein_vector=protein_vector.squeeze()
        protein_vector=F.adaptive_max_pool1d(protein_vector,output_size=1)
        # protein_vector = F.adaptive_max_pool1d(protein_vector, output_size=1)
        
        #protein_vector = protein_vector.view(protein_vector.size(0), -1)

        output_cpi_vector = torch.cat(
            (compound_vector, protein_vector.squeeze()), dim=1)
        for fc_layer in self.fc_layers:
            output_cpi_vector = fc_layer(output_cpi_vector)

        return output_cpi_vector

class DNN(nn.Module):
    def __init__(self, in_dim, out_dim, deepth=3, dropout=0.5):
        super(DNN,self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.deepth=deepth
        self.dropout=dropout
        self.MLP=nn.ModuleList()
        self._construct_DNN()

    def _construct_DNN(self):
        self.MLP.append(nn.Linear(self.in_dim,self.out_dim))
        self.MLP.append(nn.ReLU())
        for i in range(self.deepth-2):

            self.MLP.append(nn.Linear(self.out_dim,self.out_dim))
            self.MLP.append(nn.ReLU())
        self.MLP.append(nn.Linear(self.out_dim,1))
        self.MLP.append(nn.Sigmoid())
    def forward(self, features, labels):
        for layer in self.MLP:
            features=layer(features)
            
        loss=F.binary_cross_entropy(features,labels)
        return loss, features

        