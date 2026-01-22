import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from enum_holder import DataEnum
from model_config import (
    ADNI,
    AML,
    BLCA,
    BRCA,
    LIHC,
    MASKING,
    MASKING_M,
    PRAD,
    ROSMAP,
    TCGA_BRCA,
    TCGA_GBM,
    WT,
    BRCA_M,
    LGG,
    KIPAN,
    ROSMAP_M
)




def feature_level_attention(weights, dataset, train_test_val, attention_types, per_class_attention, k=100):
    label = dataset.label
    features_list = dataset.features_list
    attention_weights_omics = defaultdict(dict)
    

    if per_class_attention:
        for key, value in weights.items():
            for cl_label in torch.unique(label):
                class_label = cl_label.item()
                class_attention = value[label == class_label]
                attention_per_class = class_attention.mean(dim=0)
                try:
                    attention_per_omic1 = attention_per_class.sum(dim=1)
                except IndexError:
                    attention_per_omic1 = attention_per_class
                if attention_types == "all_features":
                    k = attention_per_omic1.shape[0]
                topk_omic1 = torch.topk(attention_per_omic1, k=k)
                top10_indices_omic1 = [val.item() for val in topk_omic1.indices]
                top10_scores_omic1 = [val.item() for val in topk_omic1.values]
                if "_" in key:
                    try:
                        fin_res = defaultdict()
                        for mod1_idx in top10_indices_omic1:
                            attention_per_omic1 = attention_per_class[mod1_idx]
                            if attention_types == "all_features":
                                k = attention_per_omic1.shape[0]
                            topk_omic2 = torch.topk(attention_per_omic1, k)
                            top5_indices_omic2 = [val.item() for val in topk_omic2.indices]
                            top5_scores_omic2 = [val.item() for val in topk_omic2.values]
                            keys_names = key.split("_")
                            omic1 = keys_names[0]
                            omic2 = keys_names[1]
                            features_name_omic1 = [features_list[omic1][i] for i in top10_indices_omic1]
                            features_name_omic2 = [features_list[omic2][i] for i in top5_indices_omic2]
                            fin_res[mod1_idx] = (features_name_omic2, top5_scores_omic2, top5_indices_omic2)
                        fin_results = (features_name_omic1, 
                                    top10_scores_omic1, 
                                    top10_indices_omic1,
                                    fin_res)
                        # top10_indices, features_names, scores, mod2_dict = fin_results
                        attention_weights_omics[key][class_label] = fin_results
                        # plot_stacked_mini_heatmaps(
                        #     top10_indices=top10_indices,
                        #     features_names=features_names,
                        #     scores=scores,
                        #     mod2_dict=mod2_dict,
                        #     omic_key=key,
                        #     class_label=0,
                        #     omic1=omic1,
                        #     omic2=omic2,
                        #     save_path=f"{save_dir}/{key}_{train_test_val}_nested_heatmap.png"
                        # )
                    except:
                        pass
                else:
                    features_name = [features_list[key][i] for i in top10_indices_omic1]
                    fin_results = (features_name, top10_scores_omic1, top10_indices_omic1)
                    attention_weights_omics[key][class_label] = fin_results
                    # plot_and_save_attention_plots(key, class_label, fin_results, save_dir)
    else:
        for key, value in weights.items():
            att = value.mean(dim=0)
            try:
                att_omics_1 = att.sum(dim=1)
            except IndexError:
                att_omics_1 = att
            if attention_types == "all_features":
                k = att_omics_1.shape[0]
            topk_omic1 = torch.topk(att_omics_1, k=k)
            top10_indices_omic1 = [val.item() for val in topk_omic1.indices]
            top10_scores_omic1 = [val.item() for val in topk_omic1.values]
            if "_" in key:
                keys_names = key.split("_")

                if len(features_list[keys_names[0]]) == att.shape[0]:
                    omic1 = keys_names[0]
                    omic2 = keys_names[1]
                else:
                    att = att.T
                    omic1 = keys_names[0]
                    omic2 = keys_names[1]
                att_combo_dict = defaultdict(list)
                for omic1_idx, omic1_val in enumerate(att):
                    for omic2_idx, omic2_val in enumerate(omic1_val):
                        att_combo_dict[
                            f"{features_list[omic1][omic1_idx]}_{features_list[omic2][omic2_idx]}"
                            ].append(omic2_val.item())
                    
                attention_weights_omics[f"{omic1}_{omic2}"] = att_combo_dict
                # try:
                #     fin_res = defaultdict()
                #     for mod1_idx in top10_indices_omic1:
                #         attention_per_omic1 = att[mod1_idx]
                #         if attention_types == "all_features":
                #             k = attention_per_omic1.shape[0]
                #         topk_omic2 = torch.topk(attention_per_omic1, k)
                #         top5_indices_omic2 = [val.item() for val in topk_omic2.indices]
                #         top5_scores_omic2 = [val.item() for val in topk_omic2.values]
                #         keys_names = key.split("_")
                #         omic1 = keys_names[0]
                #         omic2 = keys_names[1]
                #         if len(features_list[omic1]) == len(top10_indices_omic1):
                #             features_name_omic1 = [features_list[omic1][i] for i in top10_indices_omic1]
                #             features_name_omic2 = [features_list[omic2][i] for i in top5_indices_omic2]
                #         else:
                #             features_name_omic2 = [features_list[omic2][i] for i in top10_indices_omic1]
                #             features_name_omic1 = [features_list[omic1][i] for i in top5_indices_omic2]
                        
                #         fin_res[mod1_idx] = (features_name_omic2, top5_scores_omic2, top5_indices_omic2)
                #     fin_results = (features_name_omic1, 
                #                 top10_scores_omic1, 
                #                 top10_indices_omic1,
                #                 fin_res)
                #     attention_weights_omics[key] = fin_results
                # except:
                #     pass
            else:
                features_name = [features_list[key][i] for i in top10_indices_omic1]
                fin_results = (features_name, top10_scores_omic1, top10_indices_omic1)
                attention_weights_omics[key] = fin_results
    return attention_weights_omics

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


def read_omics_data_csv(gene_file_name: str, path: Path, dataset: str, labels:np.array) -> pd.DataFrame:
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
    dataset: str,
    range_data: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_idx: Tuple[np.ndarray, None] = None,
) -> Dict[str, torch.Tensor]:
    masking_dict = defaultdict()
    if dataset in [
        DataEnum.BRCA_M.name,
        DataEnum.ROSMAP_M.name,
        DataEnum.KIPAN.name,
        DataEnum.LGG.name,
    ]:
        for maskin_type, masking_index in zip(MASKING_M, [train_idx, test_idx]):
            masking_dict[maskin_type] = torch.tensor(
                [i in set(masking_index) for i in range_data]
            )
    else:
        for maskin_type, masking_index in zip(MASKING, [train_idx, val_idx, test_idx]):
            masking_dict[maskin_type] = torch.tensor(
                [i in set(masking_index) for i in range_data]
            )
    return masking_dict

def custome_train(model, layer_names):
    for layer_name in layer_names:
        for name, module in model.named_children():
            if name == layer_name:
                for param in module.parameters():
                    param.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = False
    return model


def prepare_new_data(new_dataset: Dict, dataset: Dict)-> Dict:
    new_data_col_shape = {
            omics_train_type: new_dataset.graph.nodes["patient"].data[omics_train_type].shape[1]
            for omics_train_type in new_dataset.graph.etypes
        }
    data_col_shape = {
            omics_train_type: dataset.graph.nodes["patient"].data[omics_train_type].shape[1]
            for omics_train_type in dataset.graph.etypes
        }
    
    feature_index = defaultdict()
    for omics_train_type in new_dataset.graph.etypes:
        if new_data_col_shape[omics_train_type] != data_col_shape[omics_train_type]:
            if new_data_col_shape[omics_train_type] > data_col_shape[omics_train_type]:
                omics_train_type_index = random.sample(
                    range(new_data_col_shape[omics_train_type]), data_col_shape[omics_train_type]
                    )
                omics_train_type_index = len(omics_train_type_index)
            else:
                omics_train_type_index = data_col_shape[omics_train_type] 
        else:
            omics_train_type_index = data_col_shape[omics_train_type]
        feature_index[omics_train_type] = omics_train_type_index
        
    new_data = defaultdict()
    for omics_train_type in new_dataset.graph.etypes:
        new_data_omics = new_dataset.graph.nodes["patient"].data[omics_train_type]
        if new_data_omics.shape[1] != feature_index[omics_train_type]:
            if new_data_omics.shape[1] < feature_index[omics_train_type]:
                new_data[omics_train_type] = F.pad(
                    new_data_omics, 
                    (0, abs(new_data_omics.shape[1] - feature_index[omics_train_type]))
                    , value=0)
            else:
                selected_random_index = random.sample(range(new_data_omics.shape[1]), feature_index[omics_train_type])
                new_data[omics_train_type] = new_data_omics[:, selected_random_index]
        else:
            new_data[omics_train_type] = new_data_omics
    return new_data
    

def sort_data_order(
    dataset: Any, train_data: Dict[str, torch.Tensor], forwards: bool
) -> Dict[str, torch.Tensor]:
    if dataset == DataEnum.ROSMAP.name:
        data_order = ROSMAP
    elif dataset == DataEnum.TCGA_BRCA.name:
        data_order = TCGA_BRCA
    elif dataset == DataEnum.TCGA_GBM.name:
        data_order = TCGA_GBM
    elif dataset == DataEnum.ADNI.name:
        data_order = ADNI
    elif dataset == DataEnum.AML.name:
        data_order = AML
    elif dataset == DataEnum.BLCA.name:
        data_order = BLCA
    elif dataset == DataEnum.BRCA.name:
        data_order = BRCA
    elif dataset == DataEnum.LIHC.name:
        data_order = LIHC
    elif dataset == DataEnum.PRAD.name:
        data_order = PRAD
    elif dataset == DataEnum.WT.name:
        data_order = WT
    elif dataset == DataEnum.ROSMAP_M.name:
        data_order = ROSMAP_M
    elif dataset == DataEnum.BRCA_M.name:
        data_order = BRCA_M
    elif dataset == DataEnum.KIPAN.name:
        data_order = KIPAN
    elif dataset == DataEnum.LGG.name:
        data_order = LGG
    if not forwards:
        return {f"{key}": train_data.get(f"{key}") for key in data_order[::-1]}
    elif list(train_data.keys()) == data_order:
        return train_data
    else:
        return {f"{key}": train_data.get(f"{key}") for key in data_order}

def return_dicitonaries_key(all_runs_attention_features_score: Dict) -> List:
    all_features = []
    if isinstance(all_runs_attention_features_score, list):
        all_runs_attention_features_score = {i:v for i, v in all_runs_attention_features_score}
    for i in list(all_runs_attention_features_score.keys()):
        split_name = i.split("_")
        if len(split_name) == 5:
            all_features.append("_".join(split_name[0:4]))
        elif len(split_name) == 6:
            all_features.append("_".join(split_name[0:5]))
    return np.unique(all_features).tolist()


def mrr(
    all_runs_attention_features_score: Dict[int, Dict[str, torch.Tensor]],
    omics: str,
    keys: str,
    feature_lists: List[str],
) -> Tuple[torch.Tensor, List[str]]:
    attention_stack = torch.stack(
        [value[omics] for k, value in all_runs_attention_features_score.items() if k.startswith(keys)]
    )
    feature_scores = attention_stack.mean(dim=1)

    def compute_ranks(scores: torch.Tensor) -> torch.Tensor:
        return torch.argsort(torch.argsort(-scores)) + 1
    if "_" in omics:
        ranks_first_omics = torch.stack([compute_ranks(row) for row in feature_scores])
        mrr_per_feature_first_omics = (1.0 / ranks_first_omics.float()).mean(dim=0)
        sorted_mrr_first_omics, sorted_indices_first_omics = torch.sort(mrr_per_feature_first_omics, descending=True)
        second_omics_idexes = defaultdict()
        top_mrr_features = defaultdict()
        for index in sorted_indices_first_omics:
            ranks_second_omics = torch.stack([compute_ranks(row) for row in feature_scores[:, index, : ]])
            mrr_per_feature_second_omics = (1.0 / ranks_second_omics.float()).mean(dim=0)
            sorted_mrr_second_omics, sorted_indices_second_omics = torch.sort(mrr_per_feature_second_omics, descending=True)
            second_omics_idexes[index] = sorted_indices_second_omics
            top_mrr_features[
                feature_lists[omics.split("_")[0]][index]
                ] = [
                    feature_lists[omics.split("_")[1]][sec_omic_index] 
                    for sec_omic_index in sorted_indices_second_omics[:10]
                ]
        return (
            (sorted_mrr_first_omics, sorted_indices_first_omics), 
            (sorted_mrr_second_omics, sorted_indices_second_omics), 
            top_mrr_features
            )
    else:
        ranks = torch.stack([compute_ranks(row) for row in feature_scores])
        mrr_per_feature = (1.0 / ranks.float()).mean(dim=0)
        sorted_mrr, sorted_indices = torch.sort(mrr_per_feature, descending=True)
        top_mrr_features = []
        for i in sorted_indices:
            top_mrr_features.append(feature_lists[omics][i])
        return sorted_mrr, sorted_indices, top_mrr_features


def mrr_up(
    all_runs_attention_features_score: Dict[int, Dict[str, torch.Tensor]],
    omics: str,
    feature_lists: List[str],
) -> Tuple[torch.Tensor, List[str]]:
    attention_stack = torch.stack(
        [value[omics] for k, value in all_runs_attention_features_score.items() if k.startswith(keys)]
    )
    feature_scores = attention_stack.mean(dim=1)

    def compute_ranks(scores: torch.Tensor) -> torch.Tensor:
        return torch.argsort(torch.argsort(-scores)) + 1
    if "_" in omics:
        ranks_first_omics = torch.stack([compute_ranks(row) for row in feature_scores[:, :, 0]])
        mrr_per_feature_first_omics = (1.0 / ranks_first_omics.float()).mean(dim=0)
        sorted_mrr_first_omics, sorted_indices_first_omics = torch.sort(mrr_per_feature_first_omics, descending=True)
        second_omics_idexes = defaultdict()
        for index in sorted_indices_first_omics:
            ranks_second_omics = torch.stack([compute_ranks(row) for row in feature_scores[:, index, : ]])
            mrr_per_feature_second_omics = (1.0 / ranks_second_omics.float()).mean(dim=0)
            sorted_mrr_second_omics, sorted_indices_second_omics = torch.sort(mrr_per_feature_second_omics, descending=True)
            second_omics_idexes[index] = sorted_indices_second_omics