
from model import CPI
from model import DTI,CPI_GCN,CPI_DGLLife
from dgl.data.utils import save_graphs, load_graphs
from data_loader import load_data, ExternalDataset
import utils
import numpy as np
import random
import time
import os
import torch
import torch.nn.functional as F
import argparse
import warnings
warnings.filterwarnings("ignore")
import wandb
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

def process_kg(args,train_kg,data,adj_list,degrees,use_cuda,sample_nodes=None):
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
    return g,node_id,edge_type,node_norm,grapg_data,labels,edge_norm

def graph_data_iter(batch_size,features,protein2seq):
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
            drugids.append(drugid)
        yield drugs, np.array(targets), np.array(labels), np.array(drugids)

def get_all_graph(features,protein2seq):
    drugs=list()
    targets=list()
    labels=list()
    drugids=list()
    for (drugid,targetid,label) in features:
        drugs.append(int(drugid))
        targets.append(protein2seq[int(targetid)])
        labels.append([int(label)])
        drugids.append(drugid)
    return drugs, np.array(targets), np.array(labels), np.array(drugids)

def test(model, val_dataset, protein2seq, smiles2graph):
    y_preds=[]
    y_lables=[]
    for drugs, proteins, cpi_labels,_ in graph_data_iter(1,val_dataset,protein2seq):
        y_pred=model(drugs,torch.from_numpy(np.array(proteins)),smiles2graph,True)
        y_preds.append(float(y_pred))
        y_lables.append(int(cpi_labels))
    y_preds=torch.from_numpy(np.array(y_preds))
    y_lables=torch.from_numpy(np.array(y_lables))
    val_acc, val_roc, val_pre, val_recall,val_aupr = utils.eval_cpi_2(
                    y_preds, y_lables)
    print("CPI-val | acc:{:.4f}, roc:{:.4f}, precision:{:.4f}, recall:{:.4f}, aupr:{:.4f}".
                  format( val_acc, val_roc, val_pre, val_recall, val_aupr))

def train_cpi_gcn(dataset,args):
    data = ExternalDataset('dataset/kg', dataset='bindingdb', match='celegans')
    model_path='ckl/comparision_batch_size32_lr0.005_gcn_epoch100.pkl'
    test_compounds,test_proteins,test_cpi_label,test_cpi_drugids=get_all_graph(data.test_set_gnn,data.protein2seq)
    test_cpi_label=torch.from_numpy(test_cpi_label)
    num_feature=78
    drug_size=200
    hidden_dim=200
    words=np.load('data/words_dict_{}_full_1_3.npy'.format('human'),allow_pickle=True)
    words=words.item()
    model=CPI_DGLLife(num_feature,hidden_dim,drug_size,data.word_length,device='cpu')
    model.load_state_dict(torch.load(model_path))
    model.cpu()
    model.eval()
    test_cpi_pred= model(test_compounds,torch.from_numpy(test_proteins),data.smiles2graph,True)
    results=[]
    for i in range(len(test_cpi_pred)):
        pred=float(test_cpi_pred[i])
        label=float(test_cpi_label[i])
        results.append([pred, label])
    results.sort(key=lambda x:x[0], reverse=True)
    np.save('logs/KG-MTL-S-CPI-bindingdb-prediction.npy',np.array(results))
    test_acc, test_roc, test_pre, test_recall,test_aupr = utils.eval_cpi_2(
            test_cpi_pred, test_cpi_label)
    print("Test CPI | acc:{:.4f}, roc:{:.4f}, precision:{:.4f}, recall:{:.4f}, aupr:{:.4f}".
              format(test_acc, test_roc, test_pre, test_recall,test_aupr))
    return [test_acc, test_roc, test_aupr]


def train_dti(args):
    data = ExternalDataset('dataset/kg', dataset='bindingdb')
    test_drugs, test_targets, test_dti_labels =get_dti_data(data.test_dti_set)
    test_dti_labels=torch.from_numpy(test_dti_labels)
    model = DTI(data.num_nodes,
                  200, 200, data.num_rels, 20)
    #wandb.watch(model)
    train_kg = torch.LongTensor(np.array(data.triples))   
    print('build adj and degrees....')
    if os.path.isfile('data/adj_list.npy'):
        adj_list = list(np.load('data/adj_list.npy', allow_pickle=True))
        degrees = np.load('data/degrees.npy')
    else:
        adj_list, degrees = utils.get_adj_and_degrees(data.num_nodes, train_kg)
        np.save('data/adj_list.npy', np.array(adj_list))
        np.save('data/degrees.npy', degrees)
    model_path='ckl/comparision_lr0.001_{}batch_size32_dti_single_best.pkl'.format('drugbank')
    g,node_id,edge_type,node_norm,grapg_data,labels,edge_norm=process_kg(args,train_kg,data,adj_list,degrees,use_cuda=False,sample_nodes=list(data.test_sample_nodes))
    model.load_state_dict(torch.load(model_path))
    model.cpu()
    model.eval()
    test_dti_pred, _=model(test_drugs,test_targets, g, node_id.cpu(), edge_type.cpu(), edge_norm.cpu())
    for i in range(len(test_dti_pred)):
        pred=float(test_dti_pred[i])
        label=int(test_dti_labels[i])
        results.append([pred, label])
    results.sort(key=lambda x:x[0], reverse=True)
    np.save('logs/KG-MTL-S-DTI-bindingdb-prediction.npy',np.array(results))
    test_acc, test_roc, test_pre, test_recall,test_aupr = utils.eval_cpi_2(
                test_dti_pred, test_dti_labels)
    print("DTI-test-final | acc:{:.4f}, roc:{:.4f}, precision:{:.4f}, recall:{:.4f}, aupr:{:.4f}".
                  format(test_acc, test_roc, test_pre, test_recall,test_aupr))
    #np.save('dti_single_loss_{}.npy'.format(args.dataset),np.array(loss_history))
    return [test_acc, test_roc, test_aupr]



def CPI_func(dataset): 
    return train_cpi(dataset)

def DTI_func(args):
    return train_dti(args)

def CPI_GNN_func(dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout',type=float,default=0.2,help='dropout probability')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    args1=parser.parse_args()
    print(args1)
    return train_cpi_gcn(dataset,args1)


import wandb
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float,
                        default=0.2, help='dropout probability')
    parser.add_argument('--n-hidden', type=int, default=500,
                        help='number of hidden units')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--lr_pre', type=float, default=1e-2,
                        help='learning rate of pretrain')
    parser.add_argument('--lr_dti', type=float, default=0.001,
                        help='learning rate of dti task')
    parser.add_argument('--n_bases', type=int, default=20,
                        help='number of weight blocks for each relation')
    parser.add_argument('--dti-batch-size', type=int,
                        default=128, help='batch size for dti task')
    parser.add_argument('--sample_size', type=int,
                        default=4, help='size of sample of ')
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="number of minimum training epochs")
    parser.add_argument("--eval-batch-size", type=int,
                        default=500, help="batch size when evaluating")

    parser.add_argument("--eval-protocol", type=str, default="filtered",
                        help="type of evaluation protocol: 'raw' or 'filtered' mrr")

    parser.add_argument("--regularization", type=float,
                        default=0.01, help="regularization weight")
    parser.add_argument("--grad-norm", type=float,
                        default=1.0, help="norm to clip gradient to")
    # parser.add_argument("--graph-batch-size", type=int, default=30000,
    #                     help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
                        help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
                        help="number of negative samples per positive sample")
    parser.add_argument("--edge-sampler", type=str, default="neighbor",
                        help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--graph_batch_size", type=int, default=40000)
    parser.add_argument("--rgcn_epochs", type=int,
                        default=10, help="rgcn pre-training rounds")
    parser.add_argument("--loss_lamda", type=float,
                        default=0.5, help="rgcn pre-training rounds")
    parser.add_argument('--dataset',type=str,default='bindingdb',help='dataset for dti task')
    parser.add_argument('--task',type=str,default='dti',help='[cpi, dti]')
    args = parser.parse_args()
    #celegans, human
    #CPI_func('celegans')
    results=[]
    result=CPI_GNN_func(args.dataset)
    print(result)
    # result=DTI_func(args)
    # print(result)
    # results.append(avg)
    # results.append(std)
    # np.savetxt('results/{}_{}_result_{}.txt'.format(args.task, args.dataset, 'KG-MTL-S'),
    #            np.array(results), delimiter=",", fmt='%f')

