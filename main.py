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
from data_loader import load_data
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


def main(args):
    # get dataset for gnn
    data = load_data('dataset/kg',
                     'dataset/dti_task', 'dataset/cpi_task', cpi_dataset=args.cpi_dataset, dti_dataset=args.dti_dataset, cpi_gnn=True)
    # print(len(data.compound2smiles))
    train_kg = torch.LongTensor(np.array(data.train_kg))

    val_compounds, val_proteins, val_cpi_labels, val_compoundids = get_all_graph(
        data.val_set_gnn, data.protein2seq)
    test_compounds, test_proteins, test_cpi_labels, test_compoundids = get_all_graph(
        data.test_set_gnn, data.protein2seq)
    val_cpi_labels = torch.from_numpy(val_cpi_labels)
    test_cpi_labels = torch.from_numpy(test_cpi_labels)

    val_drugs, val_targets, val_dti_labels = get_dti_data(data.val_dti_set)
    val_dti_labels = torch.from_numpy(val_dti_labels).long()
    test_drugs, test_targets, test_dti_labels = get_dti_data(data.test_dti_set)
    test_dti_labels = torch.from_numpy(test_dti_labels).long()

    drug_entities, target_entities, dti_labels = get_dti_data(
        data.train_dti_set)

    loss_model = MultiTaskLoss(2, args.shared_unit_num, args.
    embedd_dim, data.word_length, 3, 2, 0.5, data.num_nodes,
                               args.embedd_dim, args.embedd_dim, data.num_rels, args.n_bases)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        loss_model.cuda()
    dti_labels = torch.from_numpy(dti_labels).float().cuda()
    # build adj list and calculate degrees for sampling
    print('build adj and degrees....')

    if os.path.isfile('data/adj_list.npy'):
        adj_list = list(np.load('data/adj_list.npy', allow_pickle=True))
        degrees = np.load('data/degrees.npy')
    else:
        adj_list, degrees = utils.get_adj_and_degrees(data.num_nodes, train_kg)
        np.save('data/adj_list.npy', np.array(adj_list))
        np.save('data/degrees.npy', degrees)
    print('start training....')

    lr_globals = [0.001]
    batch_sizes = [32]
    # loss_lamdas=[0.25,0.5,0.75]
    shared_lrs = [0.001]
    super_params = [lr_globals, batch_sizes, shared_lrs]
    combinations = utils.lists_combination(super_params, ',')
    search_performace = dict()
    loss_history = []
    auc_history = []
    for p in combinations:
        print('params: {} training...'.format(p))
        val_dti_log = []
        search_performace[p] = dict()
        best_test_cpi_record = [0, 0]
        best_test_dti_record = [0, 0]
        best_dti_roc = 0.0
        best_cpi_roc = 0.0
        val_cpi_log = []
        epochs_his = []
        test_dti_performance = dict()
        test_cpi_performance = dict()
        l = p.strip().split(',')
        lr_g = float(l[0])  # global learning rate for each layer
        batch_size = int(l[1])  # batch_size of cpi task
        # loss_lamda=float(l[2]) # loss weight for two tasks
        shared_lr = float(l[2])  # learning rate for shared unit
        early_stop = 0
        params_list = []
        
        params = list(
            filter(lambda kv: 'shared_unit' in kv[0], loss_model.named_parameters()))

        base_params = list(
            filter(lambda kv: 'shared_unit' not in kv[0], loss_model.named_parameters()))
        for k, v in params:
            params_list += [{'params': [v], 'lr': shared_lr}]

        for k, v in base_params:
            params_list += [{'params': [v], 'lr': lr_g}]
        optimizer_global = torch.optim.Adam(params_list, lr=lr_g)

        model_path = 'ckl/lr{}_epoch{}_{}_{}_batch{}_slr{}_global_400.pkl'.format(
            lr_g, args.n_epochs, args.cpi_dataset, args.dti_dataset, batch_size, shared_lr)
        for epoch in range(args.n_epochs):
            # early stop epoch is 5
            early_stop += 1
            if early_stop >= 6:
                print(
                    'After 6 consecutive epochs, the model stops training because the performance has not improved!')
                break
            loss_model.train()
            if use_cuda:
                loss_model.cuda()
            g, node_id, edge_type, node_norm, grapg_data, labels, edge_norm = process_kg(
                args, train_kg, data, adj_list, degrees, use_cuda, sample_nodes=list(data.sample_nodes))
            loss_epoch_cpi = 0
            loss_epoch_dti = 0
            loss_epoch_total = 0
            # 修改出来dti pairs
            for (compounds, proteins, cpi_labels, compoundids) in graph_data_iter(batch_size, data.train_set_gnn, data.protein2seq):
                cpi_labels = torch.from_numpy(cpi_labels).float().cuda()
                loss_total, loss_cpi, loss_dti, cpi_pred, dti_pred, loss_params = loss_model(g, node_id, edge_type, edge_norm,
                                                                                             compounds, torch.LongTensor(proteins).cuda(), compoundids, drug_entities, target_entities, smiles2graph=data.smiles2graph, cpi_labels=cpi_labels, dti_labels=dti_labels,mode=args.loss_mode)

                loss_total.backward()
                
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer_global.step()
                optimizer_global.zero_grad()
                loss_epoch_total += loss_total
                loss_epoch_cpi += loss_cpi
                loss_epoch_dti += loss_dti
            if use_cuda:
                loss_model.cpu()
            loss_model.eval()

            cpi_pred, dti_pred = loss_model(g, node_id.cpu(), edge_type.cpu(), edge_norm.cpu(),
                                            val_compounds, torch.LongTensor(val_proteins), val_compoundids, val_drugs, val_targets, smiles2graph=data.smiles2graph, eval_=True)

            val_dti_acc, val_dti_roc, val_dti_pre, val_dti_recall, val_dti_aupr = utils.eval_cpi_2(
                dti_pred, val_dti_labels)
            
            val_acc, val_roc, val_pre, val_recall, val_aupr = utils.eval_cpi_2(
                cpi_pred, val_cpi_labels)

            test_dti_performance[str(epoch)] = [
                val_dti_acc, val_dti_roc, val_dti_pre, val_dti_recall, val_dti_aupr]
            test_cpi_performance[str(epoch)] = [
                val_acc, val_roc, val_pre, val_recall, val_aupr]
            print("Epoch {:04d}-CPI-val | acc:{:.4f}, roc:{:.4f}, precision:{:.4f}, recall:{:.4f}, aupr:{:.4f}".
                  format(epoch, val_acc, val_roc, val_pre, val_recall, val_aupr))
            val_cpi_log.append(
                [val_acc, val_roc, val_pre, val_recall, val_aupr])
            print('Epoch {:04d}-DTI-val | acc:{:.4f}, roc:{:.4f}, precision:{:.4f}, recall:{:.4f}, aupr:{:.4f}'.format(
                epoch, val_dti_acc, val_dti_roc, val_dti_pre, val_dti_recall, val_dti_aupr))
            val_dti_log.append(
                [val_dti_acc, val_dti_roc, val_dti_pre, val_dti_recall, val_dti_aupr])
            epochs_his.append(epoch)
            if best_dti_roc < val_dti_roc and best_cpi_roc < val_roc:
                early_stop = 0
                best_cpi_roc = val_roc
                best_dti_roc = val_dti_roc
                print('Best performance: CPI:{}, DTI:{}'.format(
                    best_cpi_roc, best_dti_roc))
                torch.save(loss_model.state_dict(), model_path)
                print('Best model saved!')
                # print('testing...')
                test_cpi_pred, test_dti_pred = loss_model(g, node_id.cpu(), edge_type.cpu(), edge_norm.cpu(),
                                          test_compounds, torch.LongTensor(test_proteins), test_compoundids, test_drugs,test_targets,smiles2graph=data.smiles2graph,eval_=True)
                test_dti_acc, test_dti_roc, test_dti_pre, test_dti_recall,test_dti_aupr = utils.eval_cpi_2(
                            test_dti_pred, test_dti_labels)
                test_cpi_acc, test_cpi_roc, test_cpi_pre, test_cpi_recall,test_cpi_aupr = utils.eval_cpi_2(
                test_cpi_pred, test_cpi_labels)
                # metrics={'test_dti_acc': test_dti_acc, 'test_dti_auc':test_dti_roc, 'test_dti_aupr': test_dti_aupr,     'test_cpi_acc':test_cpi_acc,'test_cpi_auc':test_cpi_roc,'test_cpi_aupr':test_cpi_aupr}
                # wandb.log(metrics)
                if best_test_cpi_record[1]<test_cpi_roc:
                    best_test_cpi_record=[test_cpi_acc, test_cpi_roc, test_cpi_aupr]
                if best_test_dti_record[1]<test_dti_roc:
                    best_test_dti_record=[test_dti_acc, test_dti_roc, test_dti_aupr]
                print("Test CPI | acc:{:.4f}, roc:{:.4f}, precision:{:.4f}, recall:{:.4f}, aupr:{:.4f}".
                  format( test_cpi_acc, test_cpi_roc, test_cpi_pre, test_cpi_recall,test_cpi_aupr))
                print('Test DTI | acc:{:.4f}, roc:{:.4f}, precision:{:.4f}, recall:{:.4f}, aupr:{:.4f}'.format(
                        test_dti_acc, test_dti_roc, test_dti_pre, test_dti_recall,test_dti_aupr))        
        
        loss_model.load_state_dict(torch.load(model_path))
        if use_cuda:
            loss_model.cpu()
        loss_model.eval()

        test_cpi_pred, test_dti_pred = loss_model(g, node_id.cpu(), edge_type.cpu(), edge_norm.cpu(),
                                                  test_compounds, torch.LongTensor(test_proteins), test_compoundids, test_drugs, test_targets, smiles2graph=data.smiles2graph, eval_=True)

        test_dti_acc, test_dti_roc, test_dti_pre, test_dti_recall, test_dti_aupr = utils.eval_cpi_2(
            test_dti_pred, test_dti_labels)
        test_cpi_acc, test_cpi_roc, test_cpi_pre, test_cpi_recall, test_cpi_aupr = utils.eval_cpi_2(
            test_cpi_pred, test_cpi_labels)
        test_dti_performance['final'] = [
            test_dti_acc, test_dti_roc, test_dti_pre, test_dti_recall, test_dti_aupr]
        test_cpi_performance['final'] = [
            test_cpi_acc, test_cpi_roc, test_cpi_pre, test_cpi_recall, test_cpi_aupr]

        # utils.Log_Writer('logs/final_unit{}_dti_{}_sulr{}_lr{}_bs{}_{}.json'.format(
        #     args.negative_sample, args.dti_dataset, shared_lr, lr_g, batch_size, args.embedd_dim), test_dti_performance)
        # utils.Log_Writer('logs/final_unit{}_cpi_{}_sulr{}_lr{}_bs{}_{}.json'.format(
        #     args.negative_sample, args.cpi_dataset, shared_lr, lr_g, batch_size, args.embedd_dim), test_cpi_performance)
        print("Test CPI | acc:{:.4f}, roc:{:.4f}, precision:{:.4f}, recall:{:.4f}, aupr:{:.4f}".
              format(test_cpi_acc, test_cpi_roc, test_cpi_pre, test_cpi_recall, test_cpi_aupr))
        print('Test DTI | acc:{:.4f}, roc:{:.4f}, precision:{:.4f}, recall:{:.4f}, aupr:{:.4f}'.format(
            test_dti_acc, test_dti_roc, test_dti_pre, test_dti_recall, test_dti_aupr))
        return [test_cpi_acc, test_cpi_roc, test_cpi_aupr], [test_dti_acc, test_dti_roc, test_dti_aupr], best_test_cpi_record, best_test_dti_record


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
                        default='celegans', help='dataset used for cpi task')
    parser.add_argument('--dti_dataset', type=str,
                        default='drugbank', help='dataset used for dti task')
    # 共用同一个shared unit layer
    parser.add_argument('--shared_unit_num', type=int,
                        default=1, help='the number of shared units')
    parser.add_argument('--embedd_dim', type=int,
                        default=128, help='the dim of embedding')
    parser.add_argument('--loss_mode', type=str,
                        default='weighted', help='the way of caculating total loss [weighted, single]')
    args = parser.parse_args()
    print(args)
    results_cpi = []
    results_dti = []
    best_results_cpi = []
    best_results_dti = []
    for i in range(10):
        cpi_r, dti_r, best_cpi_r, best_dti_r = main(args)
        results_cpi.append(cpi_r)
        results_dti.append(dti_r)
        best_results_cpi.append(best_cpi_r)
        best_results_dti.append(best_dti_r)

    avg_cpi = np.mean(np.array(results_cpi), axis=0)
    std_cpi = np.std(results_cpi, axis=0)
    print('test results: ')
    print(avg_cpi)
    avg_dti = np.mean(np.array(results_dti), axis=0)
    std_dti = np.std(np.array(results_cpi), axis=0)
    print(avg_dti)
    results_cpi.append(avg_cpi)
    results_cpi.append(std_cpi)
    results_dti.append(avg_dti)
    results_dti.append(std_dti)
    np.savetxt('results/cpi_{}_result.txt'.format(args.cpi_dataset),
               np.array(results_cpi), delimiter=",", fmt='%f')
    np.savetxt('results/dti_{}_result.txt'.format(args.dti_dataset),
               np.array(results_dti), delimiter=",", fmt='%f')
    best_avg_cpi=np.mean(np.array(best_results_cpi), axis=0)
    best_std_cpi=np.std(np.array(best_results_cpi), axis=0)
    print('best results: ')
    print(best_avg_cpi)
    best_results_cpi.append(best_avg_cpi)
    best_results_cpi.append(best_std_cpi)
    best_avg_dti=np.mean(np.array(best_results_dti), axis=0)
    best_std_dti=np.std(np.array(best_results_dti), axis=0)
    print(best_avg_dti)
    best_results_dti.append(best_avg_dti)
    best_results_dti.append(best_std_dti)
    
    np.savetxt('results/cpi_{}_best_result.txt'.format(args.cpi_dataset),
               np.array(best_results_cpi), delimiter=",", fmt='%f')
    np.savetxt('results/dti_{}_best_result.txt'.format(args.dti_dataset),
               np.array(best_results_dti), delimiter=",", fmt='%f')
    print('result saved!!!')