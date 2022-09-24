import networkx as nx
import numpy as np
import re
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx

from lib.spectralcoarsening.coarsening import spectral_graph_coarsening, multilevel_graph_coarsening, spectral_clustering_coarsening, kmeans_graph_coarsening


def get_batch_coarse_ratios(batch):
    coarse_ratios = []
    for attr in batch.__dict__:
        if "coarsened_edge_index" in attr:
            coarse_ratios.append(int(attr.split("_")[1]))
    return coarse_ratios

def get_batch_num_coarse_nodes(batch, coarse_ratios):
    num_coarse_nodes = {}
    for ratio in coarse_ratios:
        coarse_ratio_postfix = str(int(ratio*100))
        num_coarse_nodes[ratio] = getattr(batch, "num_coarse_nodes_"+coarse_ratio_postfix)
    return num_coarse_nodes


class MyData(Data):
    """ Custom Data class so that we can collate batches the right way with
    the 'coarsened_edge_index' attribute"""
    def __init__(self, data=None):
        """ From PyTorch Geometric data to MyData"""
        super(MyData, self).__init__()
        if data:
            self.edge_index = data.edge_index
            self.x = data.x
            for attr in data.__dict__:
                if "coarsened_edge_index" in attr or "num_coarse_nodes" in attr or "clusters" in attr:
                    setattr(self, attr, getattr(data, attr))
            if hasattr(data, "weight"):
                self.weight = data.weight
            self.y = data.y

    def __inc__(self, key, value):
        r"""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        if "coarsened_edge_index" in key or "clusters" in key:
            coarse_postfix = key.split("_")[-1]
            return getattr(self, "num_coarse_nodes_"+coarse_postfix)
        else:
            # Only `*index*` and `*face*` attributes should be cumulatively summed
            # up when creating batches.
            return self.num_nodes if bool(re.search('(index|face)', key)) else 0


def add_coarsened_edge_index(graph, method="sgc", coarse_ratios=[0.5], fake=False):
    """ Gets a PyTorch Gemetric Data object and adds a field with the coarsened edge list
        Available methods: 
        - "sc": from spectral clustering
        - "sgc": spectral graph coarsening from "Graph Coarsening with Preserved Spectral Properties"
        - "mlgc": multi-level graph coarsening from "Graph Coarsening with Preserved Spectral Properties"
    """
    nx_graph = to_networkx(graph, to_undirected=True)
    A = np.squeeze(np.asarray(nx.to_numpy_matrix(nx_graph)))
    new_graph = graph.clone()
    for ratio in coarse_ratios:
        num_clusters = int(graph.num_nodes * ratio)
        if num_clusters == 0:
            num_clusters = 1
        coarse_ratio_postfix = str(int(ratio*100))

        # each coarsening method returns a networkx graph, and a dictionary where keys are clusters,
        # and for each cluster there is a list with the nodes in it
        if fake: # used for not computing the coarsened versions for test graphs (instead put some fake values for the sake of batching)
            coarsened_graph = nx.generators.classic.cycle_graph(num_clusters)
            nodes_for_cluster = {i: [i] for i in range(num_clusters)}
        elif method == "sc":
            coarsened_graph, nodes_for_cluster = spectral_clustering_coarsening(A, num_clusters)
        elif method == "sgc":
            coarsened_graph, nodes_for_cluster = spectral_graph_coarsening(A, num_clusters)
        elif method == "mlgc":
            coarsened_graph, nodes_for_cluster = multilevel_graph_coarsening(A, num_clusters)
        elif method == "kmeans":
            coarsened_graph, nodes_for_cluster = kmeans_graph_coarsening(A, num_clusters)
        else:
            print("Wrong method")

        pyg_coarse_graph = from_networkx(coarsened_graph)
        setattr(new_graph, "coarsened_edge_index_"+coarse_ratio_postfix, pyg_coarse_graph.edge_index)
        setattr(new_graph, "num_coarse_nodes_"+coarse_ratio_postfix, torch.tensor(pyg_coarse_graph.num_nodes))
        nodes_to_clusters = torch.zeros(graph.num_nodes, dtype=torch.int32)
        for c in nodes_for_cluster:
            for n in nodes_for_cluster[c]:
                nodes_to_clusters[n] = c
        setattr(new_graph, "clusters_"+coarse_ratio_postfix, nodes_to_clusters)

    new_graph = MyData(new_graph)
    return new_graph


def get_preproc_coarsening_fun(method="sgc", ratio=0.5):
    def fun(graph):
        return add_coarsened_edge_index(graph, method, ratio)
    return fun
