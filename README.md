<!--
 * @Author: your name
 * @Date: 2021-05-12 05:23:23
 * @LastEditTime: 2021-05-25 02:23:41
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /kg-mtl/README.md
-->
# Knowledge Graph Enhanced Multi-Task Learning for MolecularInteraction
This is our implementation of the paper
```
Tengfei Ma, Xuan Lin, Bosheng Song, Philip S. Yu and Xiangxiang Zeng. Knowledge Graph Enhanced Multi-Task Learning for MolecularInteraction. IJCAI' 20 accepted
```
![avatar](dataset/kg-mtl.png)
Figure 1 shows the overview of KG-MTL. KG-MTL extracts the features from both knowledge graph and molecular graph in a synergistical way. Moreover, we design an effective Shared Unit that helps the model to jointly preserve the semantic relations of drug entity and the neighbor structures of compound in both levels of graphs. Extensive experiments on four real-world datasets demonstrate that our proposed KG-MTL outperforms the state-of-the-art methods on two representative molecular interaction prediction tasks: drug-target interaction (DTI) prediction and compound-proteininteraction (CPI) prediction.

## How to run ablation study
```python main_wandb.py --loss_mode single```

## How to install the envirenment
```bash
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
conda install -c dglteam dgl-cuda10.1
conda install -c rdkit rdkit==2018.09.3
pip install dgllife
```

## How to run the KG-MTL (including KG-MTL-L, KG-MTL-C)
```
usage: main.py [-h] [--dropout DROPOUT] [--n-hidden N_HIDDEN] [--gpu GPU]
               [--lr_pre LR_PRE] [--lr_dti LR_DTI] [--n_bases N_BASES]
               [--sample_size SAMPLE_SIZE] [--n-layers N_LAYERS]
               [--n-epochs N_EPOCHS] [--regularization REGULARIZATION]
               [--grad-norm GRAD_NORM] [--graph-split-size GRAPH_SPLIT_SIZE]
               [--negative-sample NEGATIVE_SAMPLE]
               [--edge-sampler EDGE_SAMPLER]
               [--graph_batch_size GRAPH_BATCH_SIZE]
               [--rgcn_epochs RGCN_EPOCHS] [--loss_lamda LOSS_LAMDA]
               [--cpi_dataset CPI_DATASET] [--dti_dataset DTI_DATASET]
               [--shared_unit_num SHARED_UNIT_NUM] [--embedd_dim EMBEDD_DIM]
               [--variant VARIANT] [--loss_mode LOSS_MODE]
```
`--loss_mode weighted` is the learning stratagies, and 'weighted' represents the certainty is used.
```bash
python main.py --loss_mode weighted --variant KG-MTL-C --gpu 1 --cpi_dataset human --dti_dataset drugcentral
python comparision.py --task cpi --dataset human
```
## Datasets
+ BindingDB https://www.bindingdb.org/bind/index.jsp
+ Human http://admis.fudan.edu.cn/negative-cpi/
+ C.elegans http://admis.fudan.edu.cn/negative-cpi/
+ DrugBank https://go.drugbank.com/releases/latest
+ DrugCentral https://drugcentral.org/
