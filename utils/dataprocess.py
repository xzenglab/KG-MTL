'''
Author: your name
Date: 2021-05-14 11:37:44
LastEditTime: 2021-05-14 12:36:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /kg-mtl/utils/dataprocess.py
'''


def clear_redudant():
    cpi_data = 'dataset/cpi_task/human_examples_global_final_1_3.tsv'
    dti_data = 'dataset/dti_task/drugcentral_dti_examples.tsv'
    target_list_file = 'dataset/dti_task/target_list_all_.tsv'
    target_dict = dict()
    with open(target_list_file, 'r') as f:
        for line in f:
            p, t, _ = line.strip().split('\t')
            target_dict[p] = t
    cpi_set = set()
    dti_set = set()
    cpi_dict = dict()
    dti_dict = dict()
    with open(cpi_data, 'r') as f:
        for line in f:
            infos = line.strip().split('\t')
            p_id = infos[0]
            if p_id not in target_dict:
                continue
            d_id = infos[3]

            cpi_set.add((target_dict[p_id], d_id, infos[6]))
            cpi_dict[(target_dict[p_id], d_id, infos[6])] = line
    with open(dti_data, 'r') as f:
        for line in f:
            infos = line.strip().split('\t')
            p_id = infos[0]
            d_id = infos[3]
            dti_set.add((p_id, d_id))
            dti_dict[(p_id, d_id)] = line

            dti_set.add((p_id, d_id, infos[6]))
            dti_dict[(p_id, d_id, infos[6])] = line
    interact_set = cpi_set.intersection(dti_set)
    print(len(interact_set))
    print(len(interact_set)/len(cpi_set))
    print(len(interact_set)/len(dti_set))
    cpi_set = cpi_set.difference(interact_set)
    dti_set = dti_set.difference(interact_set)
    with open('dataset/redundant/human_data.tsv', 'w') as f:
        for k in cpi_set:
            f.write(cpi_dict[k])
    with open('dataset/redundant/drugcentral_data.tsv', 'w') as f:
        for k in dti_set:
            f.write(dti_dict[k])


if __name__ == '__main__':
    clear_redudant()
