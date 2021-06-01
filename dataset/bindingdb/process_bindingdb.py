import pandas as pd
import numpy as np
def reconstruct_examples_bindingdb():
    data=pd.read_csv('dataset/bindingdb/BindingDB_All.tsv', sep='\t')
    print(data.columns)

if __name__=='__main__':
    reconstruct_examples_bindingdb()