import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_batch
from torch_scatter import scatter
from typing import Any, Dict, Union

from lib.data import Batch
from lib.coarsening_utils import get_batch_coarse_ratios, get_batch_num_coarse_nodes


class Loss(object):
    def on_epoch_start(self, **context):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    sum_of_diff_square = ((x1-x2)**2).sum(-1) + 1e-8
    return sum_of_diff_square.sqrt()


def moment_diff(sx1, sx2, k, og_batch, coarse_batch):
    """
    difference between moments
    """
    ss1 = scatter(sx1**k, og_batch, dim=0, dim_size=None, reduce='mean')
    ss2 = scatter(sx2**k, coarse_batch, dim=0, dim_size=None, reduce='mean')
    return l2diff(ss1,ss2)


class CentralMomentDiscrepancyLoss(Loss):
    def __init__(self, dataset_name, coarse_ratios, fine_grained=False, loss_computation_mode="og_vs_all_pairwise", cmd_coeff=1.0, unc_weight=False, coarse_pool="mean", weighted_ce=False, graph_level_cmd=False):
        """ cmd_coeff = constant to balance the regularization 
            unc_weight = wether to use the multi-task loss balancing technique from
                         "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics."
        """
        self.dataset_name = dataset_name
        self.model = None
        self.coarse_ratios = coarse_ratios
        self.fine_grained = fine_grained
        self.loss_computation_mode = loss_computation_mode
        self.cmd_coeff = cmd_coeff
        self.unc_weight = unc_weight
        self.coarse_pool = coarse_pool
        self.weighted_ce = weighted_ce
        self.graph_level_cmd = graph_level_cmd
        if weighted_ce:
            self.ce_loss = CELoss(dataset_name)

    def on_epoch_start(self, **context):
        assert 'model' in context
        self.model = context['model']

    @classmethod
    def cmd(cls, x1, x2, og_batch, coarse_batch, n_moments=5):
        """
        central moment discrepancy (cmd)
        - Zellinger, Werner et al. "Robust unsupervised domain adaptation
        for neural networks via moment alignment," arXiv preprint arXiv:1711.06114,
        2017.
        - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
        domain-invariant representation learning.", ICLR, 2017.
        """
        #print("input shapes", x1.shape, x2.shape)
        mx1 = scatter(x1, og_batch, dim=0, dim_size=None, reduce='mean')
        mx2 = scatter(x2, coarse_batch, dim=0, dim_size=None, reduce='mean')
        #print("mx* shapes should be same (batch_szie, dim)", mx1.shape, mx2.shape)
        sx1 = x1 - mx1.repeat_interleave(torch.unique(og_batch, return_counts=True)[1], dim=0)
        sx2 = x2 - mx2.repeat_interleave(torch.unique(coarse_batch, return_counts=True)[1], dim=0)
        #print("sx1, sx2 should be same size as input", sx1.shape, sx2.shape)
        dm = l2diff(mx1, mx2)
        #print("dm should have shape (batch_size,)", dm.shape)
        scms = dm
        for i in range(n_moments-1):
            # moment diff of centralized samples
            scms = scms + moment_diff(sx1, sx2, i+2, og_batch, coarse_batch)
        return scms

    def prepare_coarsened_batches(self, batch):
        coarsened_batches = {}
        for coarse_ratio in self.coarse_ratios:
            coarse_ratio_postfix = str(int(coarse_ratio*100))
            new_batch = copy.deepcopy(batch)
            new_batch.edge_index = getattr(new_batch, "coarsened_edge_index_"+coarse_ratio_postfix)
            if self.dataset_name in ["SYNTHETIC2", "brain-net"]: # unattributed graphs
                num_coarse_nodes = getattr(new_batch, "num_coarse_nodes_"+coarse_ratio_postfix)
                tot_num_coarse_nodes = torch.sum(num_coarse_nodes)
                new_batch.x = torch.ones((tot_num_coarse_nodes, 1))    
                new_batch_assignment = torch.zeros((tot_num_coarse_nodes, ), dtype=torch.int64)
                prev_idx = 0
                for i, n in enumerate(num_coarse_nodes):
                    new_batch_assignment[prev_idx:prev_idx+n] = torch.full((n, ), i, dtype=torch.int64)
                    prev_idx = prev_idx + n
                new_batch.batch = new_batch_assignment
            else: # attributed graphs
                cluster, perm = consecutive_cluster(getattr(new_batch, "clusters_"+coarse_ratio_postfix))
                if self.coarse_pool == "const":
                    num_coarse_nodes = getattr(new_batch, "num_coarse_nodes_"+coarse_ratio_postfix)
                    tot_num_coarse_nodes = torch.sum(num_coarse_nodes)
                    new_batch.x = torch.ones((tot_num_coarse_nodes, new_batch.x.shape[1]))
                elif self.coarse_pool == "mean":
                    new_batch.x = scatter(new_batch.x, cluster, dim=0, dim_size=None, reduce='mean')
                elif self.coarse_pool == "sum":
                    new_batch.x = scatter(new_batch.x, cluster, dim=0, dim_size=None, reduce='sum')
                elif self.coarse_pool == "max":
                    new_batch.x = scatter(new_batch.x, cluster, dim=0, dim_size=None, reduce='max')
                elif self.coarse_pool == "deepset":
                    deep_set = self.model.feat_aggr_net
                    new_batch.x = deep_set(new_batch.x, cluster)
                new_batch.batch = pool_batch(perm, new_batch.batch)
            coarsened_batches[coarse_ratio] = new_batch
        return coarsened_batches

    def __call__(self, batch: Batch, out: torch.Tensor):
        if self.coarse_pool == "const":
            new_batch = copy.deepcopy(batch)
            new_batch.x = torch.ones_like(batch.x)
            _ = self.model(new_batch)
        og_graph_node_embs = self.model.graph_embedder.node_embeddings
        og_graph_embs = self.model.graph_embeddings

        coarse_graphs_node_embs = {}
        coarse_graphs_embs = {}
        new_batches = self.prepare_coarsened_batches(batch)
        for coarse_ratio, new_batch in new_batches.items():
            new_batch.to(batch.x.device)
            #print("ratio", coarse_ratio)
            #print(new_batch)
            #print(new_batch.clusters_80)
            #print(len(torch.unique(new_batch.clusters_80)))
            #torch.set_printoptions(threshold=10_000)
            #print(torch.sort(torch.unique(new_batch.clusters_80))[0])
            #exit()
            _ = self.model(new_batch)
            coarse_node_embs = self.model.graph_embedder.node_embeddings
            coarse_graphs_node_embs[coarse_ratio] = coarse_node_embs
            coarse_graphs_embs[coarse_ratio] = self.model.graph_embeddings

        reg_loss = torch.tensor(0.0)
        if self.fine_grained: # apply regularization between the node embeddings for each graph and its coarsened version(s) (only for node-level cmd)

            if self.loss_computation_mode == "og_vs_all":
                x_coarse = torch.cat([coarse_graphs_node_embs[ratio] for ratio in self.coarse_ratios])

                b_idxs, counts = {}, {}
                for ratio in self.coarse_ratios:
                    b_i, c = torch.unique(new_batches[ratio].batch, return_counts=True)
                    b_idxs[ratio] = b_i
                    counts[ratio] = c
                new_idxs = {}
                new_batch_batch = []
                current_val = 0
                max_batch_val = b_idxs[self.coarse_ratios[0]].max()
                for value in range(max_batch_val+1):
                    for r in self.coarse_ratios:
                        if r not in new_idxs:
                            new_idxs[r] = []
                        pos_of_value = (b_idxs[r] == value).nonzero(as_tuple=True)[0]
                        count_of_value = counts[r][pos_of_value]
                        new_i = torch.arange(count_of_value.item()) + current_val
                        new_idxs[r].append(new_i)
                        new_batch_batch.append(torch.tensor(value).repeat(count_of_value.item()))
                        current_val = current_val + count_of_value

                for r in self.coarse_ratios:
                    new_idxs[r] = torch.cat(new_idxs[r])
                new_idxs = torch.cat([new_idxs[r] for r in self.coarse_ratios]) 

                new_batch_batch = torch.cat(new_batch_batch)
                new_coarse_node_embs = torch.zeros_like(x_coarse)
                new_coarse_node_embs = scatter(x_coarse, new_idxs, out=new_coarse_node_embs, dim=0, dim_size=None, reduce='sum')

                reg_loss = reg_loss + CentralMomentDiscrepancyLoss.cmd(og_graph_node_embs,
                                                                        new_coarse_node_embs,
                                                                        batch.batch,
                                                                        new_batch_batch).mean()
            elif self.loss_computation_mode == "og_vs_all_pairwise":
                for k, x_c in coarse_graphs_node_embs.items():
                    reg_loss = reg_loss + (CentralMomentDiscrepancyLoss.cmd(og_graph_node_embs,
                                                                            x_c,
                                                                            batch.batch,
                                                                            new_batches[k].batch).mean() / len(self.coarse_ratios))
        else: # apply regularization between node embeddings of all normal graphs, and all coarsened graphs
            if len(coarse_graphs_node_embs) > 1:
                coarse_graphs_node_embs = torch.cat([t for t in coarse_graphs_node_embs.values()])
            else:
                coarse_graphs_node_embs = list(coarse_graphs_node_embs.values())[0]
            reg_loss = CentralMomentDiscrepancyLoss.cmd(og_graph_node_embs,
                                                        coarse_graphs_node_embs,
                                                        torch.zeros_like(batch.batch),
                                                        torch.zeros(coarse_graphs_node_embs.shape[0], dtype=batch.batch.dtype))

        if self.graph_level_cmd:
            if self.fine_grained:
                if self.loss_computation_mode == "og_vs_all":
                    cg_embs = list(coarse_graphs_embs.values())
                    for i, og_graph_emb in enumerate(og_graph_embs):
                        cg_e = torch.stack([e[i, :] for e in cg_embs])
                        reg_loss += (CentralMomentDiscrepancyLoss.cmd(og_graph_emb, cg_e) / batch.num_graphs)
                elif self.loss_computation_mode == "og_vs_all_pairwise":
                    for i, og_graph_emb in enumerate(og_graph_embs):
                        for cg_embs in coarse_graphs_embs.values():
                            reg_loss += (CentralMomentDiscrepancyLoss.cmd(og_graph_emb, cg_embs[i, :]) / (batch.num_graphs*len(self.coarse_ratios)))
            else:
                cg_embs = torch.cat(list(coarse_graphs_embs.values()))
                reg_loss = reg_loss + CentralMomentDiscrepancyLoss.cmd(og_graph_embs, cg_embs)

        if self.weighted_ce:
            ce_loss = self.ce_loss(batch, out)
        else:
            ce_loss = F.cross_entropy(out, batch.y)

        if self.unc_weight:
            ce_precision = torch.exp(-self.model.ce_log_var)
            cmd_precision = torch.exp(-self.model.cmd_log_var)
            final_ce = torch.sum(ce_precision * ce_loss + self.model.ce_log_var, -1)
            final_cmd = torch.sum(cmd_precision * reg_loss + self.model.cmd_log_var, -1)
            final_loss = final_ce + final_cmd
        else:
            final_loss = ce_loss + self.cmd_coeff * reg_loss

        losses = {"tot": final_loss, "ce": ce_loss, "cmd": reg_loss}
        return losses


class SubgraphRegularizedLoss(Loss):
    def __init__(self, lam):
        self.model = None
        self.lam = lam

    def on_epoch_start(self, **context):
        assert 'model' in context
        self.model = context['model']

    def perturbe(self, x):
        new_x = torch.zeros(x.shape).to(x.device)

        # randomize the node features
        feat = np.random.choice(x.size(-1), x.size(0))
        new_x[torch.arange(x.size(0)), feat] = 1
        return new_x

    def __call__(self, batch: Batch, out: torch.Tensor):
        assert isinstance(batch, Batch)

        graphlets_repr = self.model.graph_embedder.graphlets_repr
        new_batch = Batch(
            self.perturbe(batch.x),
            batch.edge_index,
            batch.graph_has_graphlet,
            batch.graphlet_ids,
            batch.y
        )
        _ = self.model(new_batch)
        perturbed_graphlets_repr = self.model.graph_embedder.graphlets_repr
        reg_loss = torch.norm(graphlets_repr - perturbed_graphlets_repr, dim=-1, p=2).mean()
        return F.cross_entropy(out, batch.y) + self.lam * reg_loss


class CELoss(Loss):
    def __init__(self, dataset_name):
        weight = {
            "NCI1": [1 / 0.6230, 1 / 0.3770],
            "NCI109": [1 / 0.6204, 1 / 0.3796],
            "PROTEINS": [1 / 0.4197, 1 / 0.5803],
            "DD": [1 / 0.3547, 1 / 0.6453],
            "deezer_ego_nets": [1 / 0.5521, 1 / 0.4479],
            "twitch_egos": [1 / 0.3905, 1 / 0.6095],
            "IMDB-BINARY": [1 / 0.4899, 1 / 0.5101]
        }
        self.weight = weight.get(dataset_name, None)

    def on_epoch_start(self, **context):
        pass

    def __call__(self, batch: Union[Batch, PyGBatch], out: torch.Tensor):
        weight = torch.tensor(self.weight).to(out.device) if self.weight is not None else None
        return F.cross_entropy(out, batch.y, weight=weight)


class LabelSmoothingLoss(Loss):
    def __init__(self, classes: int, smoothing: float = 0.0, dim: int = -1):
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def on_epoch_start(self, **context):
        pass

    def __call__(self, batch: Batch, out: torch.Tensor):
        pred = out.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, batch.y.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class IRMLoss(Loss):
    def __init__(self, lam: float, dataset_name: str, cutoff: int = None):
        self.lam = lam
        self.cutoff = cutoff
        self.epoch = 0

        weight = {
            "NCI1": [1 / 0.6230, 1 / 0.3770],
            "NCI109": [1 / 0.6204, 1 / 0.3796],
            "PROTEINS": [1 / 0.4197, 1 / 0.5803],
            "DD": [1 / 0.3547, 1 / 0.6453],
            "deezer_ego_nets": [1 / 0.5521, 1 / 0.4479],
            "twitch_egos": [1 / 0.3905, 1 / 0.6095],
            "IMDB-BINARY": [1 / 0.4899, 1 / 0.5101]
        }
        self.weight = weight.get(dataset_name, None)

    def on_epoch_start(self, **context: Dict[str, Any]):
        assert 'epoch' in context
        self.epoch = context['epoch']

    @classmethod
    def irm_penalty(cls, out, target, weight=None):
        with torch.enable_grad():
            scale = torch.tensor(1., device=out.device, requires_grad=True)
            loss = F.cross_entropy(out * scale, target, weight=weight)
            grad = torch.autograd.grad(loss, [scale], retain_graph=True, create_graph=True)[0]
        return torch.sum(grad ** 2).item()

    def __call__(self, batch: PyGBatch, out: torch.Tensor):
        assert isinstance(batch, PyGBatch)
        _, sizes = torch.unique(batch.batch, return_counts=True)
        envs = sizes > self.cutoff if self.cutoff is not None else sizes

        weight = torch.tensor(self.weight).to(out.device) if self.weight is not None else None

        lam = self.lam if self.epoch > 100 else 1
        penalties = []
        losses = []
        for curr_env in torch.unique(envs):
            has_env = envs == curr_env
            penalties.append(IRMLoss.irm_penalty(out[has_env], batch.y[has_env], weight=weight))
            losses.append(F.cross_entropy(out[has_env], batch.y[has_env], weight=weight))

        return (sum(losses) / len(losses) + lam * sum(penalties) / len(penalties)) / lam
