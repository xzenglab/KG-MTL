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