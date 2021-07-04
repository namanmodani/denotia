# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:52:19 2021

@author: Vishnu
"""

import glob
import networkx as nx 
import pandas as pd 
import os 
import numpy as np
import torch

def Variables_run(edge_file):

    node250_df = pd.read_csv(r"C:\Users\Vishnu\Documents\ijhdfbgjws\node\freesurfer_thickness_fsaverage_smoothing10_size250_edgeweight_manhattan_graynet-nodes1.csv")
    node250 = node250_df.iloc[:, 2:]
    node250 = node250.to_numpy()
    dat250 = node250_df.set_index('~id').to_dict('index').items()

    x250 = glob.glob(edge_file)



    p250 = []
    k250 = []

    for count in range(len(x250)):
        f = pd.read_csv(x250[count], error_bad_lines=False, encoding='latin1')
        f.fillna(0.0001)

        if len(f) == 674541:
            G = nx.from_pandas_edgelist(f, '~from', '~to', edge_attr=True)
            G.add_nodes_from(dat250)
            p250.append(nx.convert_matrix.to_numpy_array(G, weight = 'd3:double'))
            k250.append(x250[count])
            
    
    adj = (np.array(p250))
    edge_index = pd.read_excel( r"C:\Users\Vishnu\Documents\COOsparse1.xlsx", engine='openpyxl')
    
    X = torch.tensor(node250, dtype=torch.float32)
    adj = torch.tensor(adj, dtype=torch.float32)

    return adj, X