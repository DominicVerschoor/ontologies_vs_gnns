"""
R-GCN and GraphSAGE node classifiers, kept structurally parallel so the only
difference between them is the convolution. Both take the whole graph as a
Data object (full-batch, see gnn_tools.train_model) carrying the node features built by
f1_node_data.node_features: a feature key and a scalar value per node, plus the
degree feature.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import RGCNConv, SAGEConv


class NodeEncoder(nn.Module):
    """
    Node input features: x = key_emb[key] + value * value_emb[key], then the
    degree feature is appended as one extra column.

    The key is a node's rdf:type set, a class IRI, a literal datatype, or one
    distinct string; value is the standardised number of a numeric/boolean/date
    literal and 0 otherwise. So equal keys share a learned vector, and numeric
    literals of one datatype lie on a learned line ordered by their value.
    """

    def __init__(self, num_keys, feature_dim):
        super().__init__()
        self.key_emb = nn.Embedding(num_keys, feature_dim)
        self.value_emb = nn.Embedding(num_keys, feature_dim)

    def forward(self, batch):
        x = self.key_emb(batch.key_id) + batch.value.unsqueeze(1) * self.value_emb(batch.key_id)
        return torch.cat([x, batch.degree_feat.unsqueeze(1)], dim=1)


class RGCNModel(nn.Module):
    def __init__(self, num_keys, feature_dim, hidden_channels_list, num_relations,
                 num_bases=None, out_channels=1, dropout=0.2):
        super().__init__()
        self.encoder = NodeEncoder(num_keys, feature_dim)
        dims = [feature_dim + 1] + hidden_channels_list + [out_channels]
        self.convs = nn.ModuleList(
            RGCNConv(dims[i], dims[i + 1], num_relations=num_relations, num_bases=num_bases)
            for i in range(len(dims) - 1)
        )
        self.dropout = dropout

    def forward(self, batch):
        x = self.encoder(batch)
        edge_index, edge_type = batch.edge_index.contiguous(), batch.edge_type.contiguous()
        for conv in self.convs[:-1]:
            x = F.relu(self._conv(conv, x, edge_index, edge_type))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self._conv(self.convs[-1], x, edge_index, edge_type)

    @staticmethod
    def _conv(conv, x, edge_index, edge_type):
        """
        One RGCNConv under gradient checkpointing: with bases, RGCNConv loops over the
        relations and keeps an N x in_dim activation per relation for backward (~3.7 GB per
        256-d layer on Pure SULO, so 3 layers do not fit in 12 GB). Checkpointing keeps only
        the layer's input and recomputes the layer during backward, so logits and gradients
        are unchanged. The conv has no randomness; dropout stays outside.
        """
        return checkpoint(conv, x, edge_index, edge_type, use_reentrant=False)


class GraphSAGEModel(nn.Module):
    """
    num_relations/num_bases: accepted but unused -- kept only so this
    constructor matches RGCNModel's shape for shared tooling. GraphSAGE
    ignores relation types (batch.edge_type is never read).

    Each layer is the GraphSAGE-mean aggregator, W_l * mean_{u in N(v)} h_u + b + W_r * h_v
    (project=False: neighbours are averaged raw, not first passed through a learned ReLU layer).
    Pre-specified, not tuned (user, 2026-10-25): it is the official implementation's
    MeanAggregator (MeanPoolingAggregator is the projected variant), PyG's SAGEConv default, and
    what PyG's ogbn-products and OGB's ogbl-collab GraphSAGE baselines use. project=True and
    aggr='max' are left to a separate sensitivity analysis.
      https://github.com/williamleif/GraphSAGE/blob/master/graphsage/aggregators.py
      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html
      https://github.com/pyg-team/pytorch_geometric/blob/2.3.0/examples/ogbn_products_sage.py
      https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/gnn.py
    """

    def __init__(self, num_keys, feature_dim, hidden_channels_list, num_relations=None,
                 num_bases=None, out_channels=1, aggr='mean', dropout=0.5):
        super().__init__()
        self.encoder = NodeEncoder(num_keys, feature_dim)
        dims = [feature_dim + 1] + hidden_channels_list + [out_channels]
        self.convs = nn.ModuleList(
            SAGEConv(dims[i], dims[i + 1], aggr=aggr, project=False)
            for i in range(len(dims) - 1)
        )
        self.dropout = dropout

    def forward(self, batch):
        x = self.encoder(batch)
        edge_index = batch.edge_index.contiguous()
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)
