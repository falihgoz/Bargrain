import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
    
class Bargrain(torch.nn.Module):
    def __init__(self, num_nodes,  hyperparam, device=None, tc_length=0):
        super().__init__()
        self.f1 = hyperparam.f1
        self.f2 = hyperparam.f2
        self.num_nodes = num_nodes
        self.conv1 = GCNConv(num_nodes, self.f1)
        self.conv2 = GCNConv(self.f1, self.f2)
        self.dropout = hyperparam.dropout
        self.device = device

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot

        # Generate off-diagonal interaction graph
        off_diag = np.ones([self.num_nodes, self.num_nodes])
        rel_rec = np.array(encode_onehot(
            np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(
            np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(device)
        self.rel_send = torch.FloatTensor(rel_send).to(device)
        self.fc_cat = nn.Linear(tc_length * 2, 2) #cobre 300

        self.fcn = nn.Sequential(
            nn.Linear(self.f2 * 2 * num_nodes, 256),
            nn.LeakyReLU(negative_slope=hyperparam.negative_slope),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=hyperparam.negative_slope),
            nn.Linear(32, 2)
        )

    def forward(self, data):
        x, edge_index, edge_weight, t = data.x, data.edge_index, data.edge_attr, data.t
        bz = t.shape[0]//self.num_nodes
        t = t.reshape((bz, -1, t.shape[1]))
        subjects_x = x.reshape((bz, -1, x.shape[1]))

        final_x_op = torch.empty(0, self.f2).to(self.device)
        for i in range(bz):
            subject_t = t[i]
            subject_x = subjects_x[i]
            receivers = torch.matmul(self.rel_rec, subject_t)
            senders = torch.matmul(self.rel_send, subject_t)
            subject_t = torch.cat([senders, receivers], dim=1)
            subject_t = torch.relu(subject_t)
            subject_t = self.fc_cat(subject_t)

            optimal_adj = torch.nn.functional.gumbel_softmax(
                subject_t, hard=True)

            optimal_adj = optimal_adj[:, 0].clone().reshape(self.num_nodes, -1)
            row, col = torch.where(optimal_adj != 0)
            optimal_edge_index = torch.stack([row, col], dim=0)
            x_op = self.conv1(subject_x, optimal_edge_index, None)
            x_op = F.relu(x_op)
            x_op = F.dropout(x_op, p=self.dropout, training=self.training)
            x_op = self.conv2(x_op.float(), optimal_edge_index, None)
            final_x_op = torch.cat((final_x_op, x_op), dim=0)

        x_corr = self.conv1(x, edge_index, None)
        x_corr = F.relu(x_corr)
        x_corr = F.dropout(x_corr, p=self.dropout, training=self.training)
        x_corr = self.conv2(x_corr.float(), edge_index, None)
    
        concatenated_embedding = torch.cat((final_x_op, x_corr), dim=1)

        concatenated_embedding = concatenated_embedding.reshape(
            (concatenated_embedding.shape[0]//self.num_nodes, -1))
        concatenated_embedding = F.dropout(
            concatenated_embedding, p=self.dropout, training=self.training)

        concatenated_embedding = self.fcn(concatenated_embedding.float())
        return concatenated_embedding, edge_index, optimal_edge_index