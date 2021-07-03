def process_drugbank4covid(dataset='drugbank'):
    drugs=dict()
    if dataset=='drugbank':
        filename='dataset/dti_task/final_dti_example.tsv'
    elif dataset=='human':
        filename='dataset/cpi_task/human_examples_final.tsv'
    with open(filename, 'r') as f:
        for line in f:
            infos=line.strip().split('\t')
            drug=infos[3]
            if drug not in drugs:
                drugs[drug]=[infos[3], infos[4], infos[5]]
    gene={'TNF-alpha':['Gene::7124', '483', 'MSTESMIRDVELAEEALPKKTGGPQGSRRCLFLSLFSFLIVAGATTLFCLLHFGVIGPQREEFPRDLSLISPLAQAVRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL'], 'IL-6':['Gene::3569','1754','MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIDKQIRYILDGISALRKETCNKSNMCESSKEALAENNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQM']}
    print('number of drugs:', len(drugs))
    for g in gene:
        with open('dataset/covid19/covid19_{}_{}'.format(dataset, g), 'w') as f:
            for d in drugs:
                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(gene[g][0],gene[g][1],gene[g][2],drugs[d][0],drugs[d][1],drugs[d][2],1))

if __name__=='__main__':
    process_drugbank4covid()