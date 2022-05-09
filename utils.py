'''
@Author: tengfei ma
@Date: 2020-05-17 09:42:20
LastEditTime: 2021-05-17 04:59:22
LastEditors: Please set LastEditors
@Description: Toolkits of rgcn model and data processing
@FilePath: /Multi-task-pytorch/utils.py
'''
import numpy as np
from sklearn.utils import shuffle
import torch
import dgl
#import matplotlib.pyplot as plt
from rdkit import Chem
import networkx as nx
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score,precision_recall_curve,auc
from sklearn.model_selection import train_test_split
#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################


def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i, triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_edge_neighborhood_without(adj_list, degrees, n_triplets, sample_size):
    edges = np.zeros((sample_size), dtype=np.int32)

    # initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        
        #在此处修改，选择所有的compound节点
        
        
        
        weights = sample_counts * seen
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            # 将孤立节点的权重置为0，也就是不对其进行采样
            weights[np.where(sample_counts == 0)] = 0
        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
            
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges

def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size,compound_ids):
    """Sample edges by neighborhool expansion.
    This guarantees that the sampled edges form a connected graph, which
    may help deeper GNNs that require information from more than one hop.
    """
    if not compound_ids:
        return sample_edge_neighborhood_without(adj_list, degrees, n_triplets, sample_size)

    if sample_size<len(compound_ids):
        print('sample_size must be larger than count of compounds for CPI task')
        exit(0)
    compound_ids=np.array(compound_ids)
    edges = np.zeros((sample_size), dtype=np.int32)

    # initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])
    l=len(compound_ids)
    for i in range(0, sample_size):
        
        #在此处修改，选择所有的compound节点
        
        
        if i <l:
            chosen_vertex=compound_ids[i]
            if seen[chosen_vertex]:
                continue
        else:
            weights = sample_counts * seen

            if np.sum(weights) == 0:
                weights = np.ones_like(weights)
                # 将孤立节点的权重置为0，也就是不对其进行采样
                weights[np.where(sample_counts == 0)] = 0
            probabilities = (weights) / np.sum(weights)
            chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        
        chosen_adj_list = adj_list[chosen_vertex]
        #print('sample {}, nodeid: {}'.format(i,chosen_vertex))
        seen[chosen_vertex] = True
        
        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges


def sample_edge_uniform(adj_list, degrees, n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)


def generate_sampled_graph_and_labels(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate,  sampler="uniform",compound_ids=None):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform(
            adj_list, degrees, len(triplets), sample_size)
    elif sampler == "neighbor":
        edges = sample_edge_neighborhood(
            adj_list, degrees, len(triplets), sample_size,compound_ids)
    else:
        raise ValueError(
            "Sampler type must be either 'uniform' or 'neighbor'.")

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = np.array(edges).transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size )
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
    # print("# sampled nodes: {}".format(len(uniq_v)))
    # print("# sampled edges: {}".format(len(src) * 2))
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels

# 为矩阵进行正则化，此处为拉普拉斯矩阵


def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm


def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    if isinstance(triplets,tuple):
        src,rel,dst=triplets
    else:
        src, rel, dst = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    # src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    # rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    # print("# nodes: {}, # edges: {}, # relations: {}".format(
    #     num_nodes, len(src), len(rel)))
    return g, rel.astype('int64'), norm.astype('int64')


def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm.float()
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']


def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    print("Test graph:")
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))


def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

#######################################################################
#
# Utility functions for evaluations (raw)
#
#######################################################################


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def perturb_and_get_raw_rank(embedding, w, a, r, b, test_size, batch_size=100):
    """ Perturb one element in the triplets
    """
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = embedding[batch_a] * w[batch_r]
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2)  # size: D x E x 1
        emb_c = embedding.transpose(0, 1).unsqueeze(1)  # size: D x 1 x V
        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c)  # size D x E x V
        score = torch.sum(out_prod, dim=0)  # size E x V
        score = torch.sigmoid(score)
        target = b[batch_start: batch_end]
        ranks.append(sort_and_rank(score, target))
    return torch.cat(ranks)

# return MRR (raw), and Hits @ (1, 3, 10)


def calc_raw_mrr(embedding, w, test_triplets, hits=[], eval_bz=100):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        # perturb subject
        ranks_s = perturb_and_get_raw_rank(
            embedding, w, o, r, s, test_size, eval_bz)
        # perturb object
        ranks_o = perturb_and_get_raw_rank(
            embedding, w, s, r, o, test_size, eval_bz)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1  # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()

#######################################################################
#
# Utility functions for evaluations (filtered)
#
#######################################################################


def filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_o = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider an object if it is part of a triplet to filter
    for o in range(num_entities):
        if (target_s, target_r, o) not in triplets_to_filter:
            filtered_o.append(o)
    return torch.LongTensor(filtered_o)


def filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_s = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider a subject if it is part of a triplet to filter
    for s in range(num_entities):
        if (s, target_r, target_o) not in triplets_to_filter:
            filtered_s.append(s)
    return torch.LongTensor(filtered_s)


def evaluate(y_pred, labels):
    roc_auc = roc_auc_score(labels, y_pred)
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    recall = recall_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    acc = accuracy_score(labels, y_pred)
    return roc_auc, recall, precision, acc


def perturb_o_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter):
    """ Perturb object in the triplets
    """
    num_entities = embedding.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_o = filter_o(triplets_to_filter, target_s,
                              target_r, target_o, num_entities)
        target_o_idx = int((filtered_o == target_o).nonzero())
        emb_s = embedding[target_s]
        emb_r = w[target_r]
        emb_o = embedding[filtered_o]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_o_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)


def perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter):
    """ Perturb subject in the triplets
    """
    num_entities = embedding.shape[0]
    ranks = []
    for idx in range(test_size):
        
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_s = filter_s(triplets_to_filter, target_s,
                              target_r, target_o, num_entities)
        target_s_idx = int((filtered_s == target_s).nonzero())
        emb_s = embedding[filtered_s]
        emb_r = w[target_r]
        emb_o = embedding[target_o]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_s_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)


def calc_filtered_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits=[]):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        triplets_to_filter = torch.cat(
            [train_triplets, valid_triplets, test_triplets]).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        print('Perturbing subject...')
        ranks_s = perturb_s_and_get_filtered_rank(
            embedding, w, s, r, o, test_size, triplets_to_filter)
        print('Perturbing object...')
        ranks_o = perturb_o_and_get_filtered_rank(
            embedding, w, s, r, o, test_size, triplets_to_filter)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1  # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()

#######################################################################
#
# Main evaluation function
#
#######################################################################


def calc_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits=[], eval_bz=100, eval_p="filtered"):
    if eval_p == "filtered":
        mrr = calc_filtered_mrr(
            embedding, w, train_triplets, valid_triplets, test_triplets, hits)
    else:
        mrr = calc_raw_mrr(embedding, w, test_triplets, hits, eval_bz)
    return mrr


def show_distribution_dataset(final_dti, save_fold):
    smiles_lens = dict()
    seq_lens = dict()
    smiles_set = set()
    seq_set = set()
    with open(final_dti, 'r') as f:
        for line in f:
            l = line.strip().split('\t')
            smiles_set.add(l[5].strip())
            seq_set.add(l[2].strip())

    for s in smiles_set:
        smiles_len = len(s)
        if smiles_len in smiles_lens:
            smiles_lens[smiles_len] += 1
        else:
            smiles_lens[smiles_len] = 1
    for s in seq_set:
        seq_len = len(s)
        if seq_len in seq_lens:
            seq_lens[seq_len] += 1
        else:
            seq_lens[seq_len] = 1
    # smiles distribution show
    smiles_lens = sorted(smiles_lens.items(), key=lambda x: x[0])
    X = [i for i, _ in smiles_lens]
    Y = [i for _, i in smiles_lens]
    plt.bar(X, Y)
    plt.ylabel('number of smiles')
    plt.xlabel('length of smiles')
    plt.title('length distribution of SMILES')
    plt.savefig(save_fold+'smiles_distribution.png')
    plt.cla()
    # sequence distribution show
    seq_lens = sorted(seq_lens.items(), key=lambda x: x[0])
    X = [i for i, _ in seq_lens]
    Y = [i for _, i in seq_lens]
    plt.bar(X, Y)
    plt.ylabel('number of sequence')
    plt.xlabel('length of sequence')
    plt.title('length distribution of sequence')
    plt.savefig(save_fold+'sequence_distribution.png')
    plt.close()

def eval_cpi(y_pred,labels):
    labels=np.array(labels.detach().cpu())
    y_pred=np.array(y_pred.detach().cpu())
    y_pred_labels=y_pred.argmax(axis=1)
    y_score=np.array([y_pred[i,index] for i,index in enumerate(y_pred_labels)])
    acc=accuracy_score(labels,y_pred_labels)
    #roc_score=roc_auc_score(labels,y_score)
    roc_score=roc_auc_score(labels,y_score)
    pre_score=precision_score(labels,y_pred_labels)
    recall=recall_score(labels,y_pred_labels)
    pr,re,_=precision_recall_curve(labels,y_score,pos_label=1)
    aupr=auc(re,pr)
    return acc,roc_score,pre_score,recall,aupr

def eval_cpi_2(y_pred,labels):
    labels=np.array(labels.detach().cpu())
    y_pred=np.array(y_pred.detach().cpu())
    #y_pred_labels=y_pred.argmax(axis=1)
    y_pred_labels=np.array([0 if i<0.5 else 1 for i in y_pred])
    #y_score=np.array([y_pred[i,index] for i,index in enumerate(y_pred_labels)])
    acc=accuracy_score(labels,y_pred_labels)
    roc_score=roc_auc_score(labels,y_pred)
    pre_score=precision_score(labels,y_pred_labels)
    recall=recall_score(labels,y_pred_labels)
    pr,re,_=precision_recall_curve(labels,y_pred,pos_label=1)
    aupr=auc(re,pr)

    return acc,roc_score,pre_score,recall,aupr

def draw_line(x,y,filename,x_label=None,y_label=None,labels=None):
    plt.xlabel=x_label
    plt.ylabel=y_label
    classes=x.shape[1]
    for i in range(classes):
        plt.plot(y,x[:,i],label=labels[i])
    plt.legend()
    plt.savefig('logs/{}.png'.format(filename))
    plt.cla()


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# convert smiles string to graph
def smiles2graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None,None,None
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  

def Log_Writer(filename, content):
    f=open(filename,'a')
    f.write(str(content)+'\n')
    f.close()

# possible combination
def lists_combination(lists, code=''):
    '''code is seperator for every element'''
    try:
        import reduce
    except:
        from functools import reduce
        
    def myfunc(list1, list2):
        return [str(i)+code+str(j) for i in list1 for j in list2]
    return reduce(myfunc, lists)
def StratifiedSplit(dataset, train_valid_test=0.2, valid_test=0.5, seed=2021):
    classes_dict=dict()
    split_dict=dict()
    for sample in dataset:
        label=sample[2]
        if label not in classes_dict:
            classes_dict[label]=list()
            classes_dict[label].append(sample)
        else:
            classes_dict[label].append(sample)
    train=list()
    valid=list()
    test=list()
    for c in classes_dict:
        samples=classes_dict[c]
        c_train,c_valid=train_test_split(samples, test_size=train_valid_test, random_state=seed, shuffle=True)
        c_valid,c_test=train_test_split(c_valid, test_size=valid_test, random_state=seed, shuffle=True)
        train.extend(c_train)
        valid.extend(c_valid)
        test.extend(c_test)
    train=shuffle(train, random_state=seed)
    valid=shuffle(valid, random_state=seed)
    test=shuffle(test, random_state=seed)
    return train, valid, test

if __name__ == "__main__":
    show_distribution_dataset(
        'dataset/dti_task/final_dti.tsv', 'dataset/images/')
