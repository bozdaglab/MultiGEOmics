import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch

from enum_holder import DataEnum
from model_config import (
    ADNI,
    AML,
    BLCA,
    BRCA,
    LIHC,
    MASKING,
    PRAD,
    ROSMAP,
    TCGA_BRCA,
    TCGA_GBM,
    WT,
)


def read_csv(path: Path, dataset: str, name: str) -> Union[torch.Tensor, pd.DataFrame]:
    list_label = pd.read_csv(f"{path}/{dataset}/{name}.csv", header=None).values
    try:
        return torch.tensor([label[0] for label in list_label]).long()
    except:
        return list_label


def read_pkl(path: Path, dataset: str, name: str) -> torch.Tensor:
    with open(path / dataset / f"{name}.pkl", "rb") as file:
        labels = pickle.load(file)
    return torch.from_numpy(labels)


def read_omics_data_pkl(gene_file_name: str, path: Path) -> pd.DataFrame:
    with open(path / f"{gene_file_name}.pkl", "rb") as file:
        row_features = pickle.load(file)
    return pd.DataFrame(row_features)


def read_omics_data_csv(gene_file_name: str, path: Path, dataset: str) -> pd.DataFrame:
    data = pd.read_csv(path / f"{gene_file_name}.csv")
    if dataset in [DataEnum.AML.name, DataEnum.LIHC.name]:
        feature_to_drop = "index"
        if "Unnamed: 0" in data.columns:
            data.rename(columns={"Unnamed: 0": feature_to_drop}, inplace=True)
    elif dataset in [DataEnum.BLCA.name, DataEnum.BRCA.name, DataEnum.PRAD.name]:
        feature_to_drop = "Case_ID"
    elif dataset == DataEnum.WT.name:
        feature_to_drop = "sample_id"
    return data.drop(feature_to_drop, axis=1)


def read_omics_train_test_data_csv(
    gene_file_name: str, test_train: str, path: Path
) -> pd.DataFrame:
    features_name = pd.read_csv(
        path / f"{gene_file_name}.csv", delimiter=",", header=None
    ).values
    row_features = pd.read_csv(
        path / f"{gene_file_name}_{test_train}.csv", delimiter=",", header=None
    ).values
    if gene_file_name == "expression":
        col_name = [gene[0].split(".")[0] for gene in features_name]
    elif gene_file_name in ["meth", "mirna"]:
        col_name = [gene[0] for gene in features_name]
    return pd.DataFrame(row_features, columns=col_name)


def masking(
    range_data: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, torch.Tensor]:
    masking_dict = defaultdict()
    for maskin_type, masking_index in zip(MASKING, [train_idx, val_idx, test_idx]):
        masking_dict[maskin_type] = torch.tensor(
            [i in set(masking_index) for i in range_data]
        )
    return masking_dict


def sort_data_order(
    args: Any, train_data: Dict[str, torch.Tensor], forwards: bool
) -> Dict[str, torch.Tensor]:
    if args.dataset == DataEnum.ROSMAP.name:
        data_order = ROSMAP
    elif args.dataset == DataEnum.TCGA_BRCA.name:
        data_order = TCGA_BRCA
    elif args.dataset == DataEnum.TCGA_GBM.name:
        data_order = TCGA_GBM
    elif args.dataset == DataEnum.ADNI.name:
        data_order = ADNI
    elif args.dataset == DataEnum.AML.name:
        data_order = AML
    elif args.dataset == DataEnum.BLCA.name:
        data_order = BLCA
    elif args.dataset == DataEnum.BRCA.name:
        data_order = BRCA
    elif args.dataset == DataEnum.LIHC.name:
        data_order = LIHC
    elif args.dataset == DataEnum.PRAD.name:
        data_order = PRAD
    elif args.dataset == DataEnum.WT.name:
        data_order = WT
    if not forwards:
        return {f"{key}": train_data.get(f"{key}") for key in data_order[::-1]}
    elif list(train_data.keys()) == data_order:
        return train_data
    else:
        return {f"{key}": train_data.get(f"{key}") for key in data_order}


def mrr(
    all_runs_attention_features_score: Dict[int, Dict[str, torch.Tensor]],
    omics: str,
    feature_lists: List[str],
) -> Tuple[torch.Tensor, List[str]]:
    attention_stack = torch.stack(
        [value[omics] for k, value in all_runs_attention_features_score.items()]
    )
    feature_scores = attention_stack.mean(dim=1)

    def compute_ranks(scores: torch.Tensor) -> torch.Tensor:
        return torch.argsort(torch.argsort(-scores)) + 1

    ranks = torch.stack([compute_ranks(row) for row in feature_scores])
    mrr_per_feature = (1.0 / ranks.float()).mean(dim=0)
    sorted_mrr, sorted_indices = torch.sort(mrr_per_feature, descending=True)
    top_mrr_features = []
    for i in sorted_indices:
        top_mrr_features.append(feature_lists[i])
    return sorted_mrr, sorted_indices, top_mrr_features
