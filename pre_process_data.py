import pickle
from collections import defaultdict
from itertools import islice, product
from pathlib import Path
from typing import Dict, List, Optional, Union

import dgl
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from dgl.data import DGLDataset
from dgl.heterograph import DGLHeteroGraph
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix

from enum_holder import DataEnum, SimilarityEnum
from helper import (
    read_csv,
    read_omics_data_csv,
    read_omics_data_pkl,
    read_omics_train_test_data_csv,
    read_pickle,
    read_pkl,
)
from model_config import (
    ADNI,
    AML,
    BLCA,
    BRCA,
    LIHC,
    PRAD,
    ROSMAP,
    TCGA_BRCA,
    TCGA_GBM,
    WT,
)


class MultiOmicsData(DGLDataset):
    def __init__(
        self,
        path: Path,
        folder_name: str,
        file_name: str,
        force_reload: bool,
        similarity_metrix: str,
        device: torch.device,
    ):
        self.path = path
        self.folder_name = folder_name
        self.device = device
        self.path_to_save = path / folder_name / f"{file_name}_{similarity_metrix}.bin"
        self.similarity_metrix = similarity_metrix
        super().__init__(name=folder_name, force_reload=force_reload)

    def read_data(self) -> Dict[str, Union[pd.DataFrame, torch.Tensor, Dict[str, str]]]:
        if self.folder_name == DataEnum.TCGA_BRCA.name:
            self.omics_type = TCGA_BRCA
        elif self.folder_name == DataEnum.TCGA_GBM.name:
            self.omics_type = TCGA_GBM
        elif self.folder_name == DataEnum.ADNI.name:
            self.omics_type = ADNI
        elif self.folder_name == DataEnum.ROSMAP.name:
            self.omics_type = ROSMAP
        elif self.folder_name == DataEnum.AML.name:
            self.omics_type = AML
        elif self.folder_name == DataEnum.BLCA.name:
            self.omics_type = BLCA
        elif self.folder_name == DataEnum.BRCA.name:
            self.omics_type = BRCA
        elif self.folder_name == DataEnum.LIHC.name:
            self.omics_type = LIHC
        elif self.folder_name == DataEnum.PRAD.name:
            self.omics_type = PRAD
        elif self.folder_name == DataEnum.WT.name:
            self.omics_type = WT
        else:
            raise ValueError
        if self.folder_name in [DataEnum.ROSMAP.name, DataEnum.TCGA_BRCA.name]:
            train_test_data = {
                f"{omic}_{train_type}": read_omics_train_test_data_csv(
                    gene_file_name=omic,
                    test_train=train_type,
                    path=self.path / self.folder_name,
                )
                for omic, train_type in product(self.omics_type, ["train", "test"])
            }
            train_test_data["label_train"] = read_csv(
                path=self.path, dataset=self.folder_name, name="labels_tr"
            )
            train_test_data["label_test"] = read_csv(
                path=self.path, dataset=self.folder_name, name="labels_te"
            )
        elif self.folder_name in [
            DataEnum.AML.name,
            DataEnum.BLCA.name,
            DataEnum.BRCA.name,
            DataEnum.LIHC.name,
            DataEnum.PRAD.name,
            DataEnum.WT.name,
        ]:
            train_test_data = {
                f"{omic}": read_omics_data_csv(
                    gene_file_name=omic,
                    path=self.path / self.folder_name,
                    dataset=self.folder_name,
                )
                for omic in self.omics_type
            }
            labels = read_csv(path=self.path, dataset=self.folder_name, name="labels")
            train_test_data["label"] = torch.tensor(
                [int(i[1]) for i in islice(labels, 1, None)]
            )
        else:
            train_test_data = {
                f"{omic}": read_omics_data_pkl(
                    gene_file_name=omic, path=self.path / self.folder_name
                )
                for omic in self.omics_type
            }
            train_test_data["label"] = read_pkl(
                path=self.path, dataset=self.folder_name, name="labels"
            )

        if len([i for i in train_test_data if i.startswith(self.omics_type[0])]) > 1:
            train_test_data = self.combine_data(train_test_data)
        train_test_data["features"] = {
            key: data.columns.tolist()
            for key, data in train_test_data.items()
            if key != "label"
        }
        return train_test_data

    def combine_data(
        self, train_test_data: Dict
    ) -> Dict[str, Union[pd.DataFrame, torch.Tensor]]:
        combine_data = defaultdict()
        for omi_type in self.omics_type:
            combine_data[omi_type] = pd.concat(
                [
                    omic_value
                    for omic_type, omic_value in train_test_data.items()
                    if omic_type.startswith(omi_type)
                ]
            )
        combine_data["label"] = torch.concat(
            [
                label_value
                for label_type, label_value in train_test_data.items()
                if label_type.startswith("label")
            ]
        )
        return combine_data

    def process(self):
        train_test_data = self.read_data()
        edge_train_test_data = {
            f"edges_{omic}": define_graph(
                similarity_type=self.similarity_metrix, data=train_test_data[f"{omic}"]
            )
            for omic in self.omics_type
        }

        graph_dict = {
            (f"patient", f"{omic}", f"patient"): (
                edge_train_test_data[f"edges_{omic}"][0],
                edge_train_test_data[f"edges_{omic}"][1],
            )
            for omic in self.omics_type
        }

        self.graph = dgl.heterograph(graph_dict).to(self.device)
        for omic in self.omics_type:
            self.graph.nodes[f"patient"].data[f"{omic}"] = (
                torch.from_numpy(train_test_data[f"{omic}"].values)
                .float()
                .to(self.device)
            )

        self.graph.shape = {
            key: val.shape
            for key, val in islice(train_test_data.items(), 0, len(train_test_data) - 1)
        }
        self.graph.omics_shapes = {
            key: val.shape
            for key, val in islice(train_test_data.items(), 0, len(train_test_data) - 1)
        }
        self.graph.label = train_test_data["label"].to(self.device)
        self.graph.num_patients = train_test_data[
            list(train_test_data.keys())[0]
        ].shape[0]
        self.graph.num_class = len(torch.unique(self.graph.label))
        self.graph.features_list = train_test_data["features"]

    def save(self) -> DGLHeteroGraph:
        with open(self.path_to_save, "wb") as file:
            pickle.dump(self.graph, file)
        return self.graph

    def load(self) -> DGLHeteroGraph:
        with open(self.path_to_save, "rb") as file:
            self.graph = pickle.load(file)
        return self.graph

    def has_cache(self) -> bool:
        return self.path_to_save.exists()

    def __getitem__(self) -> DGLHeteroGraph:
        return self.graph


class ToyData(DGLDataset):
    def __init__(self, path, folder_name, save_file, force_reload):
        self.path = path
        self.path_to_save = path / folder_name / save_file
        super().__init__(name=folder_name, force_reload=force_reload)

    def read_data(self):
        self.cna = read_pickle(path=self.path, dataset=self.folder_name, name="cna")
        self.cna_edges = read_pickle(
            path=self.path, dataset=self.folder_name, name="edges_cna"
        )
        self.exp = read_pickle(path=self.path, dataset=self.folder_name, name="exp")
        self.exp_edges = read_pickle(
            path=self.path, dataset=self.folder_name, name="edges_exp"
        )
        self.label = read_pickle(
            path=self.path, dataset=self.folder_name, name="labels"
        )
        self.train_mask, self.validation_mask = read_pickle(
            path=self.path, dataset=self.folder_name, name="mask_values"
        )

    def process(self):
        self.read_data()
        graph_dict = {
            ("patient", "expression", "patient"): (
                torch.tensor(self.exp_edges["Var1"].tolist()),
                torch.tensor(self.exp_edges["Var2"].tolist()),
            ),
            ("patient", "cna", "patient"): (
                torch.tensor(self.cna_edges["Var1"].tolist()),
                torch.tensor(self.cna_edges["Var2"].tolist()),
            ),
        }
        self.graph = dgl.heterograph(graph_dict)
        self.graph.nodes["patient"].data["expression"] = torch.from_numpy(
            self.exp.values
        ).float()
        self.graph.nodes["patient"].data["cna"] = torch.from_numpy(
            self.cna.values
        ).float()
        self.graph.nodes["patient"].data["train_mask"] = torch.from_numpy(
            np.isin(range(self.exp.shape[0]), self.train_mask)
        )
        self.graph.nodes["patient"].data["validation_mask"] = torch.from_numpy(
            np.isin(range(self.exp.shape[0]), self.validation_mask)
        )
        self.graph.label = self.label
        self.graph.train_mask = self.train_mask
        self.graph.validation_mask = self.validation_mask
        self.graph.in_feats = self.graph.nodes["patient"].data["expression"].shape[1]
        self.graph.hidden_feats = self.graph.in_feats
        self.graph.num_patients = (
            self.graph.nodes["patient"].data["expression"].shape[0]
        )
        self.graph.out_feats = int(self.graph.in_feats / 2)
        self.graph.num_class = len(np.unique(self.label))

    def save(self):
        with open(self.path_to_save, "wb") as file:
            pickle.dump(self.graph, file)
        return self.graph

    def load(self):
        with open(self.path_to_save, "rb") as f:
            self.graph = pickle.load(f)
        return self.graph

    def has_cache(self):
        return self.path_to_save.exists()

    def __getitem__(self):
        return self.graph


def cosine_distance_torch(
    x1: torch.Tensor, x2: Optional[Union[torch.Tensor]] = None, eps: float = 1e-8
) -> torch.Tensor:
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def graph_from_dist_tensor(
    dist: torch.Tensor, parameter: float, self_dist: bool = True
) -> torch.Tensor:
    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
    return g


def to_sparse(x: torch.Tensor) -> torch.Tensor:
    x_typename = torch.typename(x).split(".")[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return torch.sparse_coo_tensor(indices, values, x.size())


def gen_adj_mat_tensor(
    data: torch.Tensor, parameter: float, metric: str = "cosine"
) -> torch.Tensor:
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1 - dist
    else:
        raise NotImplementedError
    adj = adj * g
    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    return adj


def cal_adj_mat_parameter(
    edge_per_node: int, data: torch.Tensor, metric: str = "cosine"
) -> float:
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(
        dist.reshape(
            -1,
        )
    ).values[edge_per_node * data.shape[0]]
    return np.ndarray.item(parameter.data.cpu().numpy())


def make_adj(data: pd.DataFrame) -> torch.Tensor:
    df = torch.tensor(data.values)
    param = cal_adj_mat_parameter(3, df, metric="cosine")
    adj = gen_adj_mat_tensor(df, param, metric="cosine")
    return make_graphs(adj=adj)


def make_graphs(adj: torch.Tensor) -> torch.Tensor:
    adj = to_sparse(adj)
    adj = adj._indices()
    adj_mtx = to_scipy_sparse_matrix(adj).toarray()
    return from_scipy_sparse_matrix(scipy.sparse.coo_matrix(adj_mtx))[0]


def construct_graphs(construct_graph: torch.Tensor) -> torch.Tensor:
    construct_graph_average = torch.tensor(construct_graph.mean())
    adj = (construct_graph <= construct_graph_average).float()
    return make_graphs(adj)


def define_graph(similarity_type: str, data: pd.DataFrame) -> torch.Tensor:
    if similarity_type == SimilarityEnum.cosine.name:
        construct_graph = torch.tensor(cosine_similarity(data))
        similarity_matrix = construct_graphs(construct_graph)
    elif similarity_type == SimilarityEnum.euclidean.name:
        construct_graph = torch.tensor(euclidean_distances(data))
        similarity_matrix = construct_graphs(construct_graph)
    elif similarity_type == SimilarityEnum.coeff.name:
        construct_graph = torch.tensor(np.corrcoef(data))
        similarity_matrix = construct_graphs(construct_graph)
    elif similarity_type == SimilarityEnum.spearmanr.name:
        construct_graph, _ = torch.tensor(spearmanr(data))
        similarity_matrix = construct_graphs(construct_graph)
    elif similarity_type == SimilarityEnum.knngraph.name:
        construct_graph, _ = torch.tensor(
            kneighbors_graph(data, n_neighbors=5, metric="cosine", mode="connectivity")
        )
        similarity_matrix = construct_graphs(construct_graph)
    elif similarity_type == SimilarityEnum.knn.name:
        knn = NearestNeighbors(n_neighbors=5, metric="cosine")
        knn.fit(data)
        construct_graph, _ = torch.tensor(knn.kneighbors(data))
        similarity_matrix = construct_graphs(construct_graph)
    elif similarity_type == SimilarityEnum.diff.name:
        similarity_matrix = make_adj(data=data)
    return similarity_matrix
