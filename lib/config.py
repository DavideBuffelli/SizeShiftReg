from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any, List


class ModelName(Enum):
    KaryGNN = "KaryGNN"
    KaryRPGNN = "KaryRPGNN"
    GraphletCounting = "GraphletCounting"
    GNN = "GNN"
    RPGNN = "RPGNN"

    def __str__(self):
        return self.name


@dataclass
class BatchNormConfig:
    presence: bool
    affine: bool


@dataclass
class DeepSetAlternateTraining:
    presence: bool
    deepset_first: bool
    period_length: int


@dataclass
class Config:
    """
    Attributes:

        dataset_name: Name of the dataset to be used. The datasets currently
            supported are the ones in `torch_geometric.datasets.TUDataset`

        model: Name of the model to be used. Should be present in the
            enumeration present in `models.ModelName`

        num_splits: Number of splits in the k_fold. This is used to split
            the dataset in three parts: train, validation, test by
            `load_data.ksplit`

        seed: Seed for reproducibility

        batch_size

        lr: Learning rate

        num_layers: Number of GNN recursion layers in all models that use a
            GNN. In the case of models.GraphletCounting it is the number of
            Linear layers to apply to the count vector

        mlp_num_hidden: Number of hidden layers in an MLP. It should always
            be >= 1. If a MLP is instantiated with mlp_num_hidden = 1 then
            the model is [Linear, Activation, Linear]. Look at
            `models.build_mlp` for further information

        mlp_hidden_dim: Dimensionality of the hidden layer of a MLP. For
            simplicity also the dimensionality of the output
            of the MLP (indeed `self.vertex_embed_dim` returns the same value).

        graph_embed_dim: Dimensionality of the graph representation. This is
            needed only when `self.vertex_embed_dim` is not specified, since in
            that case the graph representation dimension depends on it.

        graphlet_size

        dev: device where most of the computation should be carried out

        num_epochs

        data_dir: Directory where torch_geometric.datasets.TUDataset should
            store the data. Additionally this directory is used for ESCAPE
            temporary results
    """
    dataset_name: str
    model: ModelName
    num_gpus: Optional[int] = 0
    output_dir: Optional[str] = "run"
    num_splits: int = 1
    split: int = 1
    gnn_type: str = "gin"
    seed: int = 42
    batch_size: int = 4096
    lr: float = 0.001
    num_layers: int = 1
    mlp_num_hidden: int = 1
    mlp_hidden_dim: int = 64
    graph_embed_dim: int = 64
    jk: bool = True
    graph_pooling: str = "sum"
    graphlet_size: int = 5
    irm: Optional[float] = None
    cutoff: Optional[float] = None
    reg_const: Optional[float] = None
    ################################################# CMD
    early_stopping_metric: Optional[str] = "val/accuracy" # "val/accuracy" or "val/loss"
    num_repeat_exp: Optional[int] = 1 #(number of times to repeat experimetn with different seeds)
    cmd_reg: Optional[bool] = False
    graph_level_cmd_reg: Optional[bool] = False
    coarsening_method: Optional[str] = "sgc"
    coarse_ratios: Optional[List[float]] = field(default_factory=lambda: [0.5])
    fine_grained_cmd: Optional[bool] = False
    cmd_loss_computation_mode: Optional[str] = "og_vs_all_pairwise" # "og_vs_all_pairwise" or "og_vs_all"
    cmd_coeff: Optional[float] = None
    unc_weight: Optional[bool] = False
    normalize_emb: Optional[bool] = False
    gnn_batch_norm: Optional[bool] = True
    coarse_pool: Optional[str] = None # "const","mean","max","sum","deepset"
    deepset_alternate_training: Optional[DeepSetAlternateTraining] = DeepSetAlternateTraining(False, True, 50)
    weighted_ce: Optional[bool] = False
    only_ce_for_early_stopping: Optional[bool] = False
    #################################################
    label_smooth: Optional[float] = None
    only_common: bool = False
    num_epochs: int = 500
    data_dir: str = "data"
    train_size: int = 80
    classifier_h_dim: Optional[int] = 512
    classifier_num_hidden: int = 0
    num_out: int = 3
    classifier_dropout: float = 0.0
    batch_norm: BatchNormConfig = BatchNormConfig(True, False)
    synthetic2_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_samples": 100,
            "sizes_train": [70, 80],
            "sizes_test": [140],
            "targets": [0.2, 0.5, 0.8],
        }
    )
    synthetic2single_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_samples": 100,
            "sizes_train": [80],
            "sizes_test": [140],
            "targets": [0.2, 0.5, 0.8],
        }
    )
    synthetic3_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_samples": 100,
            "sizes_train": [[10, 10]],
            "sizes_test": [[20, 20]],
            "in_probs": [0.2, 0.2],
            "feat_probs_train": [[0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1]],
            "feat_probs_test": [[0.1, 0.9, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9]],
            "targets": [0.1, 0.3],
        }
    )
    synthetic3multi_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_samples": 100,
            "sizes_train": [[10, 10], [7, 7]],
            "sizes_test": [[20, 20]],
            "in_probs": [0.2, 0.2],
            "feat_probs_train": [[0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1]],
            "feat_probs_test": [[0.1, 0.9, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9]],
            "targets": [0.1, 0.3],
        }
    )
    brain_params: Dict[str, Any] = field(
        default_factory=lambda: {"seed": 42, "val_size": 20, "test_size": 20, "p": 0.4}
    )

    @property
    def vertex_embed_dim(self) -> int:
        return self.mlp_hidden_dim

    @property
    def data_path(self) -> Path:
        if self.data_dir:
            ret = Path(self.data_dir)
        else:
            ret = Path.home() / "graphlet_data/"
        if not ret.exists():
            ret.mkdir(parents=True, exist_ok=True)
        return ret

    @property
    def data_path_complete(self) -> Path:
        return self.data_path / str(self.graphlet_size) / self.dataset_name


@dataclass
class HyperConfig:
    num_splits: List[int]
    split: List[int]
    lr: List[float]
    batch_size: List[int]
    classifier_num_hidden: List[int]
    classifier_h_dim: List[int]
    classifier_dropout: List[float]
    gpu_perc: float
    seed: List[int]


@dataclass
class HyperConfigAnyGNN(HyperConfig):
    model: List[ModelName]
    dataset_name: List[str]
    gnn_type: List[str]
    num_epochs: List[int]
    num_out: List[int]
    num_layers: List[int]
    mlp_hidden_dim: List[int]
    graph_pooling: List[str]
    jk: List[bool]
    irm: List[Optional[float]]
    cutoff: List[Optional[float]]
    reg_const: List[Optional[float]]
    label_smooth: List[Optional[float]]
    #########
    fine_grained_cmd: List[Optional[bool]]
    cmd_coeff: List[Optional[float]]
    unc_weight: List[Optional[bool]]
    normalize_emb: List[Optional[bool]]


@dataclass
class HyperConfigGraphletCounting(HyperConfig):
    graph_embed_dim: List[int]
    gc_num_layers: List[int]
