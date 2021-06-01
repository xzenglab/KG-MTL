<!--
 * @Author: your name
 * @Date: 2021-05-12 05:23:23
 * @LastEditTime: 2021-05-25 02:23:41
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /kg-mtl/README.md
-->
# Knowledge Graph Enhanced Multi-Task Learning for MolecularInteraction
The implementation of paper 'Knowledge Graph Enhanced Multi-Task Learning for MolecularInteraction'. Molecular interaction prediction is essential in various applications including drug discovery andmaterial science. The problem becomes quite challenging when the interaction is represented byunmapped relationships in molecular networks, namely molecular interaction, because it easilysuffers from (i) insufficient labeled data with many false positive samples, and (ii) ignoring alarge number of biological entities with rich information in knowledge graph. Most of the ex-isting methods cannot properly exploit the information of knowledge graph and molecule graphsimultaneously. In this paper, we propose a large-scaleKnowledgeGraph enhancedMulti-TaskLearning model, namely KG-MTL, which extracts the features from both knowledge graph andmolecular graph in a synergistical way. Moreover, we design an effectiveShared Unitthat helpsthe model to jointly preserve the semantic relations of drug entity and the neighbor structures ofcompound in both levels of graphs. Extensive experiments on four real-world datasets demon-strate that our proposed KG-MTL outperforms the state-of-the-art methods on two representativemolecular interaction prediction tasks: drug-target interaction prediction and compound-proteininteraction prediction.
## How to run ablation study
> python main_wandb.py --loss_mode single

## How to install the envirenment
```bash
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
conda install -c dglteam dgl-cuda10.1=0.4.3
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
```