import networkx as nx 
import pandas as pd 
import torch, torch_geometric
from torch_geometric.data import Dataset, DataLoader, Data
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from math import ceil
from Variables import Variables_run

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None


    def forward(self, x, adj, mask=None):

        x0 = x
        x1 = F.relu(self.conv1(x0, adj, mask))
        x2 = F.relu(self.conv2(x1, adj, mask))
        x3 = F.relu(self.conv3(x2, adj, mask))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        max_nodes = 1162
        
        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(3, 64, num_nodes)
        self.gnn1_embed = GNN(3, 64, 64, lin=False)
        
        
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        
        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, 6)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask=None)
        x = self.gnn1_embed(x, adj, mask=None)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask=None)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2 , e1 + e2


    
model = Net().to(device='cpu')
model.load_state_dict(torch.load('./denotia_v2_ad.pth'))


def pred(input_graph):
    x, adj = Variables_run(input_graph)
    prediction = model(adj, x)[0].max(dim=1)[1]
    list_of_predictions = ["normal", "ftld"]
    return (list_of_predictions[prediction])

