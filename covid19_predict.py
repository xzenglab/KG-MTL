'''
@Author: your name
@Date: 2020-05-17 13:39:08
LastEditTime: 2021-05-30 05:59:12
LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /Multi-task-pytorch/main.py
'''
import argparse

from dgl.data.utils import save_graphs, load_graphs
from model import MKDTI, MultiTaskLoss
from layer import Shared_Unit_NL
from data_loader import load_data, ExternalDataset
import utils
import random
import torch
import numpy as np
import time
import torch.nn.functional as F
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import warnings
warnings.filterwarnings("ignore")

def cpi_data_iter(batch_size, features, drug2smile=None, target2seq=None):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    features = torch.from_numpy(np.array(features))
    for i in range(0, num_examples, batch_size):
        drugs = list()
        targets = list()
        labels = list()
        drugids = list()
        j = torch.LongTensor(indices[i:min(i+batch_size, num_examples)])
        features_select = features.index_select(0, j)
        for (drugid, targetid, label) in features_select:
            drugs.append(drug2smile[int(drugid)])
            targets.append(target2seq[int(targetid)])
            labels.append([int(label)])
            drugids.append(drugid)
        yield np.array(drugs), np.array(targets), np.array(labels), np.array(drugids)


def get_data(features, drug2smiles, target2seq):
    drugs = list()
    targets = list()
    labels = list()
    drugids = list()
    for (drugid, targetid, label) in features:
        drugs.append(drug2smiles[int(drugid)])
        targets.append(target2seq[int(targetid)])
        labels.append([int(label)])
        drugids.append(drugid)

    return np.array(drugs), np.array(targets), np.array(labels), np.array(drugids)


def get_dti_data(features):
    drugs = list()
    targets = list()
    labels = list()
    drugids = list()
    for (drugid, targetid, label) in features:
        drugs.append(int(drugid))
        targets.append(int(targetid))
        labels.append([int(label)])
    return np.array(drugs), np.array(targets), np.array(labels)


def dti_data_iter(batch_size, features):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    features = torch.from_numpy(np.array(features))
    for i in range(0, num_examples, batch_size):
        drugs = list()
        targets = list()
        labels = list()
        #drugids = list()
        j = torch.LongTensor(indices[i:min(i+batch_size, num_examples)])
        features_select = features.index_select(0, j)
        for (drugid, targetid, label) in features_select:
            drugs.append(int(drugid))
            targets.append(int(targetid))
            labels.append([int(label)])
            # drugids.append(drugid)
        yield np.array(drugs), np.array(targets), np.array(labels)


def graph_data_iter(batch_size, features, protein2seq):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    features = torch.from_numpy(np.array(features))
    for i in range(0, num_examples, batch_size):
        drugs = list()
        targets = list()
        labels = list()
        drugids = list()
        j = torch.LongTensor(indices[i:min(i+batch_size, num_examples)])
        features_select = features.index_select(0, j)
        for (drugid, targetid, label) in features_select:
            drugs.append(int(drugid))
            targets.append(protein2seq[int(targetid)])
            labels.append([int(label)])
            drugids.append(int(drugid))
        yield drugs, np.array(targets), np.array(labels), np.array(drugids)


def get_all_graph(features, protein2seq):
    drugs = list()
    targets = list()
    labels = list()
    drugids = list()
    for (drugid, targetid, label) in features:
        drugs.append(int(drugid))
        targets.append(protein2seq[int(targetid)])
        labels.append([int(label)])
        drugids.append(drugid)
    return drugs, np.array(targets), np.array(labels), np.array(drugids)


def process_kg(args, train_kg, data, adj_list, degrees, use_cuda, sample_nodes=None):
    g, node_id, edge_type, node_norm, grapg_data, labels = utils.generate_sampled_graph_and_labels(
        train_kg, args.graph_batch_size, args.graph_split_size, data.num_rels, adj_list, degrees, args.negative_sample, args.edge_sampler, sample_nodes)

    #print('Done edge sampling for rgcn')
    node_id = torch.from_numpy(node_id).view(-1, 1).long()
    edge_type = torch.from_numpy(edge_type)
    edge_norm = utils.node_norm_to_edge_norm(
        g, torch.from_numpy(node_norm).view(-1, 1))
    grapg_data, labels = torch.from_numpy(
        grapg_data), torch.from_numpy(labels)
    deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
    if use_cuda:

        node_id, deg = node_id.cuda(), deg.cuda()
        edge_norm, edge_type = edge_norm.cuda(), edge_type.cuda()
        grapg_data, labels = grapg_data.cuda(), labels.cuda()
        # test_node_id,test_deg=test_node_id.cuda(),test_deg.cuda()
        # test_norm,test_rel=test_norm.cuda(),test_rel.cuda()
    return g, node_id, edge_type, node_norm, grapg_data, labels, edge_norm


def KG_MTL(args):
    # get dataset for gnn
    data = ExternalDataset('dataset/kg', dataset='TNF-alpha',match='human')
    # print(len(data.compound2smiles))
    train_kg = torch.LongTensor(np.array(data.triples))
    test_compounds, test_proteins, test_cpi_labels, test_compoundids = get_all_graph(
        data.test_set_gnn, data.protein2seq)
    test_cpi_labels = torch.from_numpy(test_cpi_labels)
    
    words=np.load('data/words_dict_{}_full_1_3.npy'.format('human'),allow_pickle=True)
    words=words.item()
    test_drugs, test_targets, test_dti_labels = get_dti_data(data.test_dti_set)
    test_dti_labels = torch.from_numpy(test_dti_labels).long()
    ### get all infos for bindingdb
    device='cuda:{}'.format(args.gpu) if args.gpu>=0 else 'cpu'
    loss_model = MultiTaskLoss(2, args.shared_unit_num, args.
    embedd_dim, len(words), 3, 2, 0.5, data.num_nodes,
                               args.embedd_dim, args.embedd_dim, data.num_rels, args.n_bases, variant=args.variant, device=device)
    
    print('build adj and degrees....')

    if os.path.isfile('data/adj_list.npy'):
        adj_list = list(np.load('data/adj_list.npy', allow_pickle=True))
        degrees = np.load('data/degrees.npy')
    else:
        adj_list, degrees = utils.get_adj_and_degrees(data.num_nodes, train_kg)
        np.save('data/adj_list.npy', np.array(adj_list))
        np.save('data/degrees.npy', degrees) 
    g, node_id, edge_type, node_norm, grapg_data, labels, edge_norm = process_kg(
            args, train_kg, data, adj_list, degrees, use_cuda=False, sample_nodes=list(data.test_sample_nodes))     
    # model_path='ckl/lr0.001_epoch100_human_drugcentral_batch32_slr0.001_128_KG-MTL.pkl'
    model_path='ckl/lr0.001_epoch22_human_drugcentral_batch32_slr0.001_128_KG-MTL.pkl'
    loss_model.load_state_dict(torch.load(model_path))
    loss_model.eval()
    test_cpi_pred, test_dti_pred = loss_model(g, node_id.cpu(), edge_type.cpu(), edge_norm.cpu(),
                                              test_compounds, torch.LongTensor(test_proteins), test_compoundids,test_drugs, test_targets, smiles2graph=data.smiles2graph, eval_=True)
    y_pred=torch.cat((test_cpi_pred, test_dti_pred), dim=1)
    y_pred=y_pred.detach().numpy()
    y_pred_labels=y_pred.argmax(axis=1)
    y_score=np.array([y_pred[i,index] for i,index in enumerate(y_pred_labels)])
    scores=[]
    with open('dataset/covid19/covid19_human_TNF-alpha','r') as f:
        for i, l in enumerate(f):
            scores.append([y_score[i],l.strip().split('\t')[3]])
    
    scores=sorted(scores, key=lambda keys:keys[0], reverse=True)
    with open('dataset/covid19/scores_human_{}_kg-mtl.tsv'.format(args.dti_dataset), 'w') as f:
        for [s, c] in scores:
            f.write('{}\t{}\n'.format(s,c))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float,
                        default=0.2, help='dropout probability')
    parser.add_argument('--n-hidden', type=int, default=500,
                        help='number of hidden units')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--lr_pre', type=float, default=0.01,
                        help='learning rate of pretrain')
    parser.add_argument('--lr_dti', type=float, default=0.001,
                        help='learning rate of dti task')
    parser.add_argument('--n_bases', type=int, default=4,
                        help='number of weight blocks for each relation')
    parser.add_argument('--sample_size', type=int,
                        default=4, help='size of sample of ')
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="number of minimum training epochs")
    parser.add_argument("--regularization", type=float,
                        default=0.01, help="regularization weight")
    parser.add_argument("--grad-norm", type=float,
                        default=1.0, help="norm to clip gradient to")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
                        help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
                        help="number of negative samples per positive sample")
    parser.add_argument("--edge-sampler", type=str, default="neighbor",
                        help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--graph_batch_size", type=int, default=40000)
    parser.add_argument("--rgcn_epochs", type=int,
                        default=0, help="rgcn pre-training rounds")
    parser.add_argument("--loss_lamda", type=float,
                        default=0.75, help="rgcn pre-training rounds")
    parser.add_argument('--cpi_dataset', type=str,
                        default='TNF-alpha', help='dataset used for cpi task')
    parser.add_argument('--dti_dataset', type=str,
                        default='TNF-alpha', help='dataset used for dti task')
    # 共用同一个shared unit layer
    parser.add_argument('--shared_unit_num', type=int,
                        default=1, help='the number of shared units')
    parser.add_argument('--embedd_dim', type=int,
                        default=128, help='the dim of embedding')
    parser.add_argument('--variant', type=str,
                        default='KG-MTL', help='[KG-MTL, KG-MTL-L, KG-MTL-C]')
    parser.add_argument('--loss_mode', type=str,
                        default='weighted', help='the way of caculating total loss [weighted, single]')
    args = parser.parse_args()
    print(args)
    results_cpi = []
    results_dti = []
    best_results_cpi = []
    best_results_dti = []
    KG_MTL(args)
    # print('cpi: ', cpi_r)
    # print('dti: ', dti_r)
