'''
@Author: tengfei ma
@Date: 2020-05-16 17:50:18
LastEditTime: 2021-05-23 02:41:38
LastEditors: Please set LastEditors
@Description: 加载RGCN数据以及DTI
@FilePath: /Multi-task-pytorch/data_loader.py
'''
import utils
from sklearn.model_selection import train_test_split,StratifiedKFold
import numpy as np
import torch
import os
import dgllife
CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25}

CHARPROTLEN = 25  # count of categories

CHARCANSMISET = {"#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
                         ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
                         "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
                         "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
                         "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
                         "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
                         "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
                         "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
                         "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
                         "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
                         "t": 61, "y": 62}

CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64  # count of labels

MAX_SMI_LEN = 200
MAX_SEQ_LEN = 1200

'''
@description: label smiles string with a fixed length
@param {type} 
@return: 
'''
def label_smiles(line, MAX_SMI_LEN, smi_ch_id):
    X = np.zeros(MAX_SMI_LEN,dtype=int)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_id[ch]
    return X.tolist()


'''
@description: label protein sequence with a fixed length
@param {type} 
@return: 
'''
def label_sequence(line, MAX_SEQ_LEN, seq_ch_id):
    X = np.zeros(MAX_SEQ_LEN,dtype=int)

    for i, ch in enumerate(line[: MAX_SEQ_LEN]):
        X[i] = seq_ch_id[ch]
    
    return X.tolist()

'''
@description: 
@param {type} 
@return: 
'''
def split_grams(seq, n=3):
    one = zip(*[iter(seq)]*n) # handle first seq
    two = zip(*[iter(seq[1:])]*n)
    three = zip(*[iter(seq[2:])]*n)

    total = [one,two,three]
    str_ngram = set()
    for ngrams in total:
        
        for ngram in  ngrams:
            str_ngram.add(''.join(ngram))
        #str_ngram.append(x)
    return list(str_ngram)
def label_sequence_by_words(seq,words_dict,max_lenght=1200):
    
    ngrams_words=split_grams(seq)
    X=np.zeros(max_lenght,dtype=int)
    for i, word in enumerate(ngrams_words[:max_lenght]):
        if word in words_dict:

            X[i]=words_dict[word]
        else:
            X[i]=words_dict['-+-']

    return X
def storeWordsIntoDict(sequences,dataset):
    if os.path.isfile('data/words_dict_{}_full_1_3.npy'.format(dataset)):
        words=np.load('data/words_dict_{}_full_1_3.npy'.format(dataset),allow_pickle=True)
        return words.item()
    words=dict()
    words['-+-']=0
    max_length=0
    print('process sequence for {}'.format(dataset))
    count=0
    lens=len(sequences)
    for seq in sequences:
        count+=1
        #print('{}/{}'.format(count, lens))
        ngram_words=split_grams(seq)
        if max_length<len(ngram_words):
            max_length=len(ngram_words)
        for w in ngram_words:
            if w not in words:
                words[w]=len(words)
    
    np.save('data/words_dict_{}_full_1_3.npy'.format(dataset),words)
    print('max words length of protein is {}'.format(max_length))
    return words
def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    #dataset_1,dataset_2=train_test_split(dataset,test_size=ratio)
    return dataset_1, dataset_2

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
class load_data():
    def __init__(self, kg_file, dti_path=None,cpi_path=None,cpi_dataset='human',dti_dataset='drugbank',cpi_gnn=False,test_model=False):
        
        #
        self.cpi_dataset=cpi_dataset
        self.dti_dataset=dti_dataset
        self.train_kg, self.num_nodes, self.num_rels = self._load_kg_data(
            kg_file)
        if test_model:
            self.train_dti_set,self.val_dti_set,self.test_dti_set,self.test_sample_nodes=self._load_dti_data(dti_path)
            if not cpi_gnn:
                self.train_cpi_set, self.val_cpi_set, self.test_cpi_set, self.compound2smiles, self.protein2seq = self._load_cpi_data(
                    cpi_path)
                for i in self.compound2smiles:
                    self.test_sample_nodes.add(int(i))
            else:
                self.train_set_gnn,self.val_set_gnn,self.test_set_gnn,self.smiles2graph,self.protein2seq,self.word_length= self._load_cpi_gnn(cpi_path)
                for i in self.smiles2graph:
                    self.test_sample_nodes.add(int(i))
            self.test_dti_global,self.test_sample_nodes=self._load_dti_global(dti_path)
            self.test_cpi_global,self.test_smiles2graph,self.test_protein2seq,self.test_word_length=self._load_cpi_global_test(cpi_path)
            for i in self.test_smiles2graph:
                self.test_sample_nodes.add(int(i))
        else:
            
            self.train_dti_set,self.val_dti_set,self.test_dti_set,self.sample_nodes=self._load_dti_data(dti_path)
            if not cpi_gnn:
                self.train_cpi_set, self.val_cpi_set, self.test_cpi_set, self.compound2smiles, self.protein2seq = self._load_cpi_data(
                    cpi_path)
                for i in self.compound2smiles:
                    self.sample_nodes.add(int(i))
            else:
                self.train_set_gnn,self.val_set_gnn,self.test_set_gnn,self.smiles2graph,self.protein2seq,self.word_length= self._load_cpi_gnn(cpi_path)
                #self.cpi_examples, self.smiles2graph,self.protein2seq,self.word_length= self._load_cpi_gnn(cpi_path)
                for i in self.smiles2graph:
                    self.sample_nodes.add(int(i))

    def _load_kg_data(self, kg_path):
        test_triples = list()
        entity2id = dict()
        relation2id = dict()
        train_triples = list()
        validate_triples = list()
        # with open('{}/drkg.tsv'.format(kg_path), 'r') as f:
        #     for line in f:
        #         head, rel, tail = line.strip().split('\t')
        #         if head not in entity2id:
        #             entity2id[head] = len(entity2id)
        #         if tail not in entity2id:
        #             entity2id[tail] = len(entity2id)
        #         if rel not in relation2id:
        #             relation2id[rel] = len(relation2id)

        with open('{}/entities.tsv'.format(kg_path),'r') as f:
            for line in f:
                e_name,e_id=line.strip().split('\t')
                entity2id[e_name]=int(e_id)
        with open('{}/relations.tsv'.format(kg_path),'r') as f:
            for line in f:
                r_name,r_id=line.strip().split('\t')
                relation2id[r_name]=int(r_id)
        num_entities = len(entity2id)
        num_rels = len(relation2id)
        
        # read kg train set
        with open('{}/drkg.tsv'.format(kg_path), 'r') as f:
            for line in f:
                head, rel, tail = line.strip().split('\t')
                train_triples.append([entity2id[head], relation2id[rel], entity2id[tail]])

        entity2id = None
        relation2id = None
        return train_triples, num_entities, num_rels

    def _load_cpi_data(self, cpi_path):
        drug2smiles = dict()
        target2seq = dict()
        examples = list()
        
        if self.cpi_dataset=='celegans':
            example_path='{}/celegan_examples_global_final_1_3.tsv'.format(cpi_path)
        else:
            example_path='{}/human_examples_global_final_1_3.tsv'.format(cpi_path)
        # if self.cpi_dataset=='drugbank':
        #     example_path='{}/final_dti_example.tsv.tsv'.format(cpi_path)
        # else:
        #     example_path='{}/drugcentral_dti_examples.tsv'.format(cpi_path)
        with open(example_path, 'r') as f:
            for line in f:
                l = line.strip().split('\t')
                drug_id = int(l[4])
                target_id = int(l[1])
                seq = l[2]
                smiles = l[5]
                label = int(l[6])
                drug2smiles[drug_id] = label_smiles(
                    smiles, MAX_SMI_LEN, CHARISOSMISET)
                target2seq[target_id] = label_sequence(
                    seq, MAX_SEQ_LEN, CHARPROTSET)
                examples.append([drug_id, target_id, label])

        #8:1:1
        train_set, test_set = train_test_split(
            examples, test_size=0.2,random_state=555)
        val_set, test_set = train_test_split(
            test_set, test_size=0.5,random_state=555)
        return train_set, val_set, test_set, drug2smiles, target2seq
        #return examples, drug2smiles, target2seq
    def _load_dti_data(self,dti_path):
        examples=list()
        sample_ndoes=set()
        
        if self.dti_dataset=='drugbank':
            example_path='{}/final_dti_example.tsv'.format(dti_path)
            # example_path='dataset/redundant/dti_data.tsv'
        elif self.dti_dataset=='drugcentral':
            example_path='{}/drugcentral_dti_examples.tsv'.format(dti_path)
        elif self.dti_dataset=='drugbank_redundant':
            example_path='dataset/redundant/dti_data.tsv'
        elif self.dti_dataset=='drugcentral_redundant':
            example_path='dataset/redundant/drugcentral_data.tsv'
        elif self.dti_dataset=='bindingdb':
            example_path='dataset/bindingdb/compound_protein_interaction.tsv'
        print(example_path)
        
        with open(example_path,'r') as f:
            for line in f:
                l=line.strip().split('\t')
                drug_entityid=int(l[4])
                target_entityid=int(l[1])
                sample_ndoes.add(drug_entityid)
                sample_ndoes.add(target_entityid)
                label=int(l[6])
                examples.append([drug_entityid,target_entityid,label])
        
        train_dti_set,test_dti_set=train_test_split(examples,test_size=0.2,random_state=3)
        val_dti_set,test_dti_set=train_test_split(examples,test_size=0.5,random_state=4)

        #train_dti_set, val_dti_set, test_dti_set=utils.StratifiedSplit(examples)

        return train_dti_set,val_dti_set,test_dti_set, sample_ndoes


    def _load_cpi_gnn(self,cpi_path):
        examples = list()
        smiles_graph=dict()
        protein2seq = dict()
        proteins_list=set()
        if self.cpi_dataset=='celegans':
            example_path='{}/celegan_examples_global_final_1_3.tsv'.format(cpi_path)
        #     #example_path_='{}/human_examples_global_final_1_3.tsv'.format(cpi_path)
        elif self.cpi_dataset=='human':
            example_path='{}/human_examples_global_final_1_3.tsv'.format(cpi_path)
        elif self.cpi_dataset=='human_redundant':
            example_path='dataset/redundant/human_data.tsv'
        elif self.cpi_dataset=='human_r':
            example_path='dataset/redundant/cpi_data.tsv'
        elif self.dti_dataset=='bindingdb':
            example_path='dataset/bindingdb/compound_protein_interaction.tsv'
        print(example_path)
        with open(example_path, 'r') as f:
            for line in f:
                l = line.strip().split('\t')
                drug_id = int(l[4])
                target_id = int(l[1])
                seq = l[2]
                smiles = l[5]
                label = int(l[6])
                proteins_list.add(seq)
                protein2seq[target_id] = seq
                if smiles not in smiles_graph:
                    c_size,features,edge_index=utils.smiles2graph(smiles)
                    if c_size is None and features is None and edge_index is None:
                        continue
                    smiles_graph[drug_id]=(c_size,features,edge_index)
                    #smiles_graph[drug_id]=dgllife.utils.smi
                examples.append([drug_id, target_id, label])

        # with open(example_path_,'r') as f:
        #     l = line.strip().split('\t')
        #     seq = l[2]
        #     proteins_list.add(seq)
            
        words_dict=storeWordsIntoDict(list(proteins_list),self.cpi_dataset)
        for p in protein2seq:
            protein2seq[p]=label_sequence_by_words(protein2seq[p],words_dict)
        #examples=shuffle_dataset
        
        # train_set,test_set=train_test_split(examples,test_size=0.2,random_state=4)
        # val_set,test_set=train_test_split(test_set,test_size=0.5,random_state=5)

        
        ### use shuffle
        train_set, val_set, test_set=utils.StratifiedSplit(examples)
        return train_set,val_set,test_set, smiles_graph,protein2seq,len(words_dict)

class ExternalDataset():
    def __init__(self, kg_path, dataset='bindingdb'):
        self.dataset=dataset
        self.triples, self.num_nodes, self.num_rels=self._load_kg_data(kg_path)
        self.test_set_gnn,self.smiles2graph,self.protein2seq,self.word_length=self._load_cpi_data()
        self.test_dti_set,self.test_sample_nodes=self._load_dti_data()
        for i in self.smiles2graph:
                    self.test_sample_nodes.add(int(i))
    def _load_kg_data(self, kg_path):
        entity2id = dict()
        relation2id = dict()
        triples = list()
        with open('{}/entities.tsv'.format(kg_path),'r') as f:
            for line in f:
                e_name,e_id=line.strip().split('\t')
                entity2id[e_name]=int(e_id)
        with open('{}/relations.tsv'.format(kg_path),'r') as f:
            for line in f:
                r_name,r_id=line.strip().split('\t')
                relation2id[r_name]=int(r_id)
        num_entities = len(entity2id)
        num_rels = len(relation2id)
        # read kg train set
        with open('{}/drkg.tsv'.format(kg_path), 'r') as f:
            for line in f:
                head, rel, tail = line.strip().split('\t')
                triples.append([entity2id[head], relation2id[rel], entity2id[tail]])

        entity2id = None
        relation2id = None
        return triples, num_entities, num_rels
    def _load_cpi_data(self):
        examples = list()
        smiles_graph=dict()
        protein2seq = dict()
        proteins_list=set()
        if self.dataset=='bindingdb':
            example_path='dataset/bindingdb/compound_protein_interaction.tsv'
        print(example_path)
        with open(example_path, 'r') as f:
            for line in f:
                l = line.strip().split('\t')
                drug_id = int(l[4])
                target_id = int(l[1])
                seq = l[2]
                smiles = l[5]
                label = int(l[6])
                proteins_list.add(seq)
                protein2seq[target_id] = seq
                if smiles not in smiles_graph:
                    c_size,features,edge_index=utils.smiles2graph(smiles)
                    if c_size is None and features is None and edge_index is None:
                        continue
                    smiles_graph[drug_id]=(c_size,features,edge_index)
                    #smiles_graph[drug_id]=dgllife.utils.smi
                examples.append([drug_id, target_id, label])
            
        words_dict=storeWordsIntoDict(list(proteins_list),'human')
        for p in protein2seq:
            protein2seq[p]=label_sequence_by_words(protein2seq[p],words_dict)
        #examples=shuffle_dataset
        
        # train_set,test_set=train_test_split(examples,test_size=0.2,random_state=4)
        # val_set,test_set=train_test_split(test_set,test_size=0.5,random_state=5)
        
        ### use shuffle
        #train_set, val_set, test_set=utils.StratifiedSplit(examples)
        return examples, smiles_graph,protein2seq,len(words_dict)
    def _load_dti_data(self):
        examples=list()
        sample_ndoes=set()
        if self.dataset=='bindingdb':
            example_path='dataset/bindingdb/compound_protein_interaction.tsv'
        print(example_path)
        
        with open(example_path,'r') as f:
            for line in f:
                l=line.strip().split('\t')
                drug_entityid=int(l[4])
                target_entityid=int(l[1])
                sample_ndoes.add(drug_entityid)
                sample_ndoes.add(target_entityid)
                label=int(l[6])
                examples.append([drug_entityid,target_entityid,label])
        
        # train_dti_set,test_dti_set=train_test_split(examples,test_size=0.2,random_state=3)
        # val_dti_set,test_dti_set=train_test_split(examples,test_size=0.5,random_state=4)

        #train_dti_set, val_dti_set, test_dti_set=utils.StratifiedSplit(examples)

        return examples, sample_ndoes

    
if __name__ == "__main__":
    # test function
    data = load_data('dataset/kg', 'dataset/dti_task')
