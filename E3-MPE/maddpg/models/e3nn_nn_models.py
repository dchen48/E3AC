from typing import Dict

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.v2106.gate_points_message_passing import MessagePassing
from e3nn.nn.models.v2106.gate_points_networks import scatter

class GraphNetWithAttributes(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        max_radius, number_of_basis,  # Edge length embedding
        num_neighbors,                # Scaling constant for MessagePassing
        num_nodes, pool_nodes=True,   # Scaling constant for node pooling
        mul=50, layers=3, lmax=2,     # Hiiden layers and their irreps
        fc_neurons = 100,             # Hiiden layers for MessagePassing
    ) -> None: 
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes

        irreps_node_hidden = o3.Irreps([
            (mul, (l, p))
            for l in range(lmax + 1)
            for p in [-1, 1]
        ])

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_node_input] + layers * [irreps_node_hidden] + [irreps_node_output],
            irreps_node_attr=irreps_node_attr,
            # irreps_edge_attr=self.irreps_edge_attr + o3.Irreps.spherical_harmonics(lmax),
            irreps_edge_attr=irreps_edge_attr,
            fc_neurons=[number_of_basis, fc_neurons],
            num_neighbors=num_neighbors,
        )

        self.irreps_node_input = self.mp.irreps_node_input
        self.irreps_node_attr = self.mp.irreps_node_attr
        self.irreps_edge_attr = self.mp.irreps_edge_attr
        self.irreps_node_output = self.mp.irreps_node_output

    def preprocess(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        # Create graph
        if 'edge_src' in data and 'edge_dst' in data:
            edge_src = data['edge_src']
            edge_dst = data['edge_dst']
        elif 'edge_index' in data:
            # edge_index = radius_graph(data['pos'], self.max_radius, batch)
            edge_src = data['edge_index'][0]
            edge_dst = data['edge_index'][1]
        else:
            print('ERROR: No edge in data')
            raise NotImplementedError

        # Edge attributes
        edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]

        if 'x' in data:
            node_input = data['x']
        elif 'node_input' in data :
            node_input = data['node_input']
        else:
            print('ERROR: No node_input in data')
            raise NotImplementedError

        node_attr = data['node_attr']
        edge_attr = data['edge_attr']

        return batch, node_input, node_attr, edge_attr, edge_src, edge_dst, edge_vec

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch, node_input, node_attr, edge_attr, edge_src, edge_dst, edge_vec = self.preprocess(data)
        del data

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis='smooth_finite',  # the smooth_finite basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        node_outputs = self.mp(node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

        if self.pool_nodes:
            return scatter(node_outputs, batch, int(batch.max()) + 1).div(self.num_nodes**0.5)
        else:
            return node_outputs
