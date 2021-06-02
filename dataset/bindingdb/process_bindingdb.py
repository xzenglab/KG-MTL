import pandas as pd
import numpy as np
import csv
def reconstruct_examples_bindingdb():
    field={
        'BindingDB Reactant_set_id':0,
        'Ligand SMILES':1,
        'Target Name Assigned by Curator or DataSource':6,
        'Target Source Organism According to Curator or DataSource':7,
        'Kd (nM)':10, #该指标指示结合亲和度
        'PubChem CID':28,
        'ChEMBL ID of Ligand' : 31,
        'ChEBI ID of Ligand' :30,
        'ZINC ID of Ligand': 35,
        'DrugBank ID of Ligand':32,
        'BindingDB Target Chain  Sequence':37,
        'UniProt (SwissProt) Primary ID of Target Chain':41
    }
    tsv_file=open('dataset/bindingdb/compound_protein_affinity.tsv','w',newline='',encoding='utf-8')
    csv_write=csv.DictWriter(tsv_file, fieldnames=list(field.keys()))
    row_count=0
    with open('dataset/bindingdb/BindingDB_All.tsv', 'r') as f:
        for line in f:
            infos=line.strip().split('\t')
            if len(infos)<=40:
                continue
            temp_dict=dict()
            if not infos[field['Kd (nM)']] or not infos[field['UniProt (SwissProt) Primary ID of Target Chain']]:
                continue
            for col in field:
                temp_dict[col]=infos[field[col]]
            row_count+=1
            csv_write.writerow(temp_dict)
    print('row count:', row_count)

def process_cpi_affinitytoInteract():
    # data=pd.read_csv('dataset/bindingdb/compound_protein_affinity.tsv')
    # data['Kd (nM)']=data['Kd (nM)'].astype('float')
    # print(len(data))
    # pos_index=data['Kd (nM)']<30
    # print('pos: ', len(pos_index))
    # neg_index=data['Kd (nM)']>=30
    # print('neg: ', len(neg_index))
    field={
        'BindingDB Reactant_set_id':0,
        'Ligand SMILES':1,
        'Target Name Assigned by Curator or DataSource':6,
        'Target Source Organism According to Curator or DataSource':7,
        'Kd (nM)':10, #该指标指示结合亲和度
        'PubChem CID':28,
        'ChEMBL ID of Ligand' : 31,
        'ChEBI ID of Ligand' :30,
        'ZINC ID of Ligand': 35,
        'DrugBank ID of Ligand':32,
        'BindingDB Target Chain  Sequence':37,
        'UniProt (SwissProt) Primary ID of Target Chain':41
    }
    with open('dataset/bindingdb/compound_protein_affinity.tsv','r') as f:
        csv_reader=csv.DictReader(f,fieldnames=list(field.keys()))
        # for line in f:
        #     infos=line.strip().split('\t')
        #     print(infos)
        pos=list()
        neg=list()
        compound_entities=dict()
        gene_entities=dict()
        uniprots=set()
        for idx, row in enumerate(csv_reader):
            if idx==0:
                continue
            # if not 'Human' in row['Target Source Organism According to Curator or DataSource']:
            #     continue
            print(row['Kd (nM)'])
            if float(row['Kd (nM)'].strip('>').strip('<'))<30:
                pos.append(row)
            else:
                neg.append(row)
        
            uniprots.add(row['UniProt (SwissProt) Primary ID of Target Chain'])
        
        print('pos: ', len(pos))
        print('neg: ', len(neg))
        protein_list=open('dataset/bindingdb/proteins.tsv','w')
        for p in uniprots:
            protein_list.write(p+'\n')
'''
original code is here: https://github.com/xiaomingaaa/drugbank/blob/master/BioMedicalKits.py
'''
def process_protein():
    #def UniprotToOtherDB(uniprot_list, savepath, filename, tran='P_ENTREZGENEID', savetype='csv'):
    uniprot_list=list()
    savepath='dataset/bindingdb/'
    filename='proteins_uniprot_geneid.tsv'
    tran='P_ENTREZGENEID'
    savetype='tsv'
    with open('dataset/bindingdb/proteins.tsv','r') as f:
        for line in f:
            uniprot_list.append(line.strip())
    file_path = '{}/{}'.format(savepath, filename)
    # uniprot api
    url = 'https://www.uniprot.org/uploadlists/'

    if isinstance(uniprot_list, list):
        param = ''
        for i in uniprot_list:
            param += ' '+i
        params = {
            'from': 'ACC+ID',
            'to': tran,
            'format': 'tab',
            'query': param
        }
    elif isinstance(uniprot_list, dict):
        param = ''
        for i in list(uniprot_list.keys()):
            param += ' '+i
        params = {
            'from': 'ACC+ID',
            'to': tran,
            'format': 'tab',
            'query': param
        }
    else:
        print('** data type is not supported **')
        return
    import urllib.request as r
    import urllib
    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = r.Request(url, data)
    pairs = list()
    try:
        with urllib.request.urlopen(req, timeout=30) as f:
            response = f.read()
        content = response.decode('utf-8')
        tt = content.strip().split('\n')
        if len(tt) > 1:
            for i in tt[1:-1]:
                info_temp = dict()
                protein, target_value = i.strip().split('\t')[:2]
                info_temp['uniprot_id'] = protein
                info_temp[tran] = target_value
                pairs.append(info_temp)
        df = pd.DataFrame(pairs)
        if savetype == 'csv':
            df.to_csv(file_path)
        elif savetype == 'excel':
            df.to_excel(file_path)
        elif savetype == 'tsv':
            df.to_csv(file_path, sep='\t', index=False, header=None)
        else:
            print('** data type is not supported **')
        # if len(tt)>1:
        #     return tt[1].strip().split('\t')[1]
    except Exception as e:
        print(e)

def map_entity_drkg():
    ### 实体 id映射表
    entities=dict()
    uniprot_gid=dict()
    with open('dataset/bindingdb/proteins_uniprot_geneid.tsv', 'r') as f:
        for line in f:
            uniprotid, g_id=line.strip().split('\t')
            uniprot_gid[uniprotid]=g_id
    with open('dataset/kg/entities.tsv', 'r') as f:
        for line in f:
            e, e_id=line.strip().split('\t')
            entities[e]=e_id
    field={
        'BindingDB Reactant_set_id':0,
        'Ligand SMILES':1,
        'Target Name Assigned by Curator or DataSource':6,
        'Target Source Organism According to Curator or DataSource':7,
        'Kd (nM)':10, #该指标指示结合亲和度
        'PubChem CID':28,
        'ChEMBL ID of Ligand' : 31,
        'ChEBI ID of Ligand' :30,
        'ZINC ID of Ligand': 35,
        'DrugBank ID of Ligand':32,
        'BindingDB Target Chain  Sequence':37,
        'UniProt (SwissProt) Primary ID of Target Chain':41
    }
    c_id={'bindingdb':'BindingDB Reactant_set_id','drugbank':'DrugBank ID of Ligand', 'pubchem':'PubChem CID', 'chebi':'ChEBI ID of Ligand', 'zinc':'ZINC ID of Ligand'}
    with open('dataset/bindingdb/compound_protein_affinity.tsv','r') as f:
        csv_reader=csv.DictReader(f,fieldnames=list(field.keys()))
        # for line in f:
        #     infos=line.strip().split('\t')
        #     print(infos)
        pos=0
        neg=0


        examples=open('dataset/bindingdb/compound_protein_interaction.tsv', 'w')
        for idx, row in enumerate(csv_reader):
            if idx==0:
                continue
            label=0
            if float(row['Kd (nM)'].strip('>').strip('<'))<30:
                label=1
            smiles=row['Ligand SMILES']
            if row['UniProt (SwissProt) Primary ID of Target Chain'] in uniprot_gid:
                gene_id=uniprot_gid[row['UniProt (SwissProt) Primary ID of Target Chain']]
            else:
                continue
            gene_sequence=row['BindingDB Target Chain  Sequence']
            if 'Gene::'+gene_id in entities:
                for col in c_id:
                    compound_id='Compound::'+col+':'+row[c_id[col]]
                    compound_id=compound_id.replace('drugbank:','')
                    if compound_id in entities:
                        examples.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('Gene::'+gene_id, entities['Gene::'+gene_id], gene_sequence, compound_id, entities[compound_id], smiles, label))
                        if label==1:
                            pos+=1
                        else:
                            neg+=1
            continue
            
        
            
        
        print('pos: ',pos)
        print('neg: ',neg)
if __name__=='__main__':
    #reconstruct_examples_bindingdb()
    #process_cpi_affinitytoInteract()
    # process_protein()
    map_entity_drkg()