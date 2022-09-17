import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("models/")
from mlp import MLP

from torch.autograd import Variable

from .cov_norm import FastMPNSPDMatrix

class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, fac_dim, rep_dim, final_dropout, learn_eps,
                 graph_pooling_type, neighbor_pooling_type, fo_type, so_type,device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.fo_type = fo_type
        self.so_type = so_type

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # for bilinear mapping second-order pooling
        self.fea_dim = input_dim + hidden_dim * (num_layers - 1)
        self.rep_dim = rep_dim
        self.fac_dim = fac_dim
        self.map_dim = self.rep_dim * self.fac_dim

        # self.graph_pool = nn.Linear(self.fea_dim, self.map_dim, bias=False)
        self.mpn_cov = FastMPNSPDMatrix()
        self.prediction = nn.Linear(self.fea_dim, output_dim)




    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        # compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                # add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                # padding, dummy data is assumed to be stored in -1
                pad.extend([-1] * (max_deg - len(pad)))

                # Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        # Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

        return Adj_block.to(self.device)

    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)

        start_idx = [0]

        # compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1. / len(graph.g)] * len(graph.g))

            else:
                ###sum pooling
                elem.extend([1] * len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]
        return pooled_rep

    def next_layer_eps(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        ###pooling neighboring nodes and center nodes altogether

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def add_gauss_noise_with_fix_snr(self,x, snr = 15):

        noise = torch.randn(x.shape[0], x.shape[1]).to(self.device)
        noise = noise - torch.mean(noise)
        signal_power = torch.norm(x - x.mean()) ** 2 / (x.shape[0]*x.shape[1])  # signal power
        noise_variance = signal_power /(10**(snr / 10))   # noise power
        noise = (torch.sqrt(noise_variance) / torch.std(noise)) * noise
        signal_noise = noise + x
        return signal_noise


    def gen_blockdiag_matrix(self, matrix_height, matrix_width, block_height, block_width):
        blockdiag_matrix = torch.zeros(matrix_height, matrix_width).to(self.device)
        list1 = list(range(0, matrix_height, block_height))
        list2 = list(range(0, matrix_width, block_width))
        for i, j in zip(list1, list2):
            blockdiag_matrix[i:i+block_height, j:j+block_width] = 1
        return blockdiag_matrix

    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)

        # graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        # list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat

        for layer in range(self.num_layers - 1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block=Adj_block)

            hidden_rep.append(h)

        hidden_rep = torch.cat(hidden_rep, 1)

        graph_sizes = [graph.node_features.size()[0] for graph in batch_graph]
        node_embeddings = torch.split(hidden_rep, graph_sizes, dim=0)

        batch_fo = torch.zeros(len(graph_sizes), self.fea_dim).to(self.device)
        batch_fo = Variable(batch_fo)

        batch_so = torch.zeros(len(graph_sizes), self.fea_dim, self.fea_dim).to(self.device)
        batch_so = Variable(batch_so)

        for g_i in range(len(graph_sizes)):
            cur_node_embeddings = node_embeddings[g_i]
            num_nodes = cur_node_embeddings.size(0)
            if self.fo_type == 0:
                cur_graph_fo = torch.mean(cur_node_embeddings, 0)
            elif self.fo_type == 1:
                cur_graph_fo = torch.sum(cur_node_embeddings, 0)
            else:
                cur_graph_fo = torch.sum(cur_node_embeddings, 0)
            if self.so_type == 0:
                mean_center_matrix = 1 / num_nodes * (torch.eye(num_nodes) - 1 / num_nodes * torch.ones(num_nodes)).to(
                    self.device)
                cur_graph_so = torch.matmul(cur_node_embeddings.t(), mean_center_matrix)
                cur_graph_so = torch.matmul(cur_graph_so, cur_node_embeddings)
            elif self.so_type == 1:
                cur_graph_so = torch.matmul(cur_node_embeddings.t(), cur_node_embeddings)
            else:
                cur_graph_so = torch.matmul(cur_node_embeddings.t(), cur_node_embeddings)

            batch_fo[g_i] = cur_graph_fo
            batch_so[g_i] = cur_graph_so

        batch_so = self.mpn_cov(batch_so)
        batch_fo = batch_fo.unsqueeze(2)
        graph_representation = batch_so.bmm(batch_fo).squeeze(2)
        graph_representation = graph_representation


        score =  F.dropout(self.prediction(graph_representation), self.final_dropout, training=self.training)

        return score
