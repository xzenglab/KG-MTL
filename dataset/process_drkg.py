'''
Author: your name
Date: 2021-05-28 13:29:18
LastEditTime: 2021-05-28 13:29:55
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /kg-mtl/dataset/process_drkg.py
'''
import random
def create_imbalance_data_human():
    proteins=dict()
    drugs=dict()
    examples=dict()
    example_writer=open('dataset/cpi_task/human_examples_global_final_full_imbalance.tsv','w')
    with open('dataset/cpi_task/human_examples_final.tsv','r') as f:
        for line in f:
            infos=line.strip().split('\t')
            if infos[6]=='1':
                infos=line.strip().split('\t')
                proteins[infos[0]]=[infos[1],infos[2]]
                drugs[infos[3]]=[infos[4],infos[5]]
                examples[(infos[0],infos[3])]=line.strip()
                example_writer.write(line)
    print('positive number: ',len(examples))
    count=len(examples)
    # for i in range(9*count):
    #     #print(i+1)
    #     d=random.choice(list(drugs.keys()))
    #     p=random.choice(list(proteins.keys()))
    #     while (p,d) in examples:
    #         d=random.choice(list(drugs.keys()))
    #         p=random.choice(list(proteins.keys()))
    #     l='{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(p,proteins[p][0],proteins[p][1],d,drugs[d][0],drugs[d][1],0)
    #     examples[(p,d)]=l
    #     example_writer.write(l)
    # example_writer.close()
    # print('total number: ', len(examples))
    for d in drugs:
        for p in proteins:
            if (p,d) not in examples:
                l='{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(p,proteins[p][0],proteins[p][1],d,drugs[d][0],drugs[d][1],0)
                examples[(p,d)]=l
                example_writer.write(l)
    example_writer.close()
    print('full number: ', len(examples))
    print('rate: ', count/len(examples))
def create_imbalance_data_drugcentral():
    proteins=dict()
    drugs=dict()
    examples=dict()
    example_writer=open('dataset/dti_task/drugcentral_examples_global_final_1_9.tsv','w')
    with open('dataset/dti_task/drugcentral_dti_examples.tsv','r') as f:
        for line in f:
            infos=line.strip().split('\t')
            if infos[6]=='1':
                infos=line.strip().split('\t')
                proteins[infos[0]]=[infos[1],infos[2]]
                drugs[infos[3]]=[infos[4],infos[5]]
                examples[(infos[0],infos[3])]=line.strip()
                example_writer.write(line)
    print('positive number: ',len(examples))
    count=len(examples)
    for i in range(9*count):
        #print(i+1)
        d=random.choice(list(drugs.keys()))
        p=random.choice(list(proteins.keys()))
        while (p,d) in examples:
            d=random.choice(list(drugs.keys()))
            p=random.choice(list(proteins.keys()))
        l='{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(p,proteins[p][0],proteins[p][1],d,drugs[d][0],drugs[d][1],0)
        examples[(p,d)]=l
        example_writer.write(l)
    example_writer.close()
    print('total number: ', len(examples))
    # for d in drugs:
    #     for p in proteins:
    #         if (p,d) not in examples:
    #             l='{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(p,proteins[p][0],proteins[p][1],d,drugs[d][0],drugs[d][1],0)
    #             examples[(p,d)]=l
    #             example_writer.write(l)
    # example_writer.close()
    print('full number: ', len(examples))
    print('rate: ', count/len(examples))


if __name__=='__main__':
    # create_imbalance_data_human()
    create_imbalance_data_drugcentral()