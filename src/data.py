from torch_geometric.data import Data
import numpy as np
from torch.utils.data import Dataset
import torch
from torch_geometric.utils import to_undirected

class Brain(Dataset):
    def __init__(self, indices,  dataset, th):
        super().__init__()
        self.tc = dataset['tc']
        self.corr_graph = dataset['corr_graph']
        self.labels = dataset['labels']

        self.nsubjects = len(indices)
        self.nrois = self.corr_graph.shape[1]
        self.ntime = self.tc.shape[2]
        self.indices = indices

        self.th = th
        self.graphs = []
        self.labels_processed = []
        count = 0

        for idx in indices:
            tc_subject = self.tc[idx]

            label = self.labels[idx]
            adj_matrix = self.corr_graph[idx]

            norm_adj_matrix = np.zeros_like(adj_matrix)
            norm_adj_matrix[adj_matrix >= self.th] = 1
            np.fill_diagonal(norm_adj_matrix, 0) 

            norm_adj_matrix = torch.tensor(norm_adj_matrix, dtype=torch.float)
            row, col = torch.where(norm_adj_matrix != 0)

            # combine the row and column indices into edge indices
            edge_index = torch.stack([row, col], dim=0)

            # convert to undirected graph by adding reverse edges
            edge_index = to_undirected(edge_index)

            row, col = edge_index
            edge_weight = adj_matrix[row, col]
            edge_weight = torch.tensor(edge_weight).double()

            t = torch.tensor(tc_subject, dtype=torch.float)
            x = torch.tensor(self.corr_graph[idx], dtype=torch.float)

            graph = Data(x=x, t=t,
                         edge_index=edge_index, edge_attr=edge_weight, y=label)

            self.graphs.append(graph)
            self.labels_processed.append(graph.y)

            count += 1
            
    def __len__(self):
        return self.nsubjects

    def __getitem__(self, index):
        return self.graphs[index]

    def __getallitems__(self):
        return self.graphs