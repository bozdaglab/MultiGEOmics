import pickle
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from enum_holder import DataEnum
from helper import feature_level_attention, masking, mrr, sort_data_order
from model import MultiGraphGCN
from model_config import RANDOM_SEEDS, TCGA_BRCA
from pre_process_data import MultiOmicsData, ToyData
from train_eval_IGCN import create_optimizer, model_evaluate, model_test, model_train


def run_model(
    config: Dict, args: Any, path: Path, random_state: int
) -> Tuple[
    float, float, float, float, torch.Tensor, Dict[str, torch.Tensor], MultiOmicsData
]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.dataset.islower():
        args.dataset = args.dataset.upper()
    if args.dataset == DataEnum.toy.name:
        dataset = ToyData(
            path=path,
            folder_name=args.dataset,
            save_file=f"{args.dataset}_data.bin",
            force_reload=True,
        )
    else:
        dataset = MultiOmicsData(
            path=path,
            folder_name=args.dataset,
            file_name=f"{args.dataset}_data",
            force_reload=True,
            similarity_metrix=config["similarity_metrix"],
            device=device,
        )

    if args.dataset in [
        DataEnum.ADNI.name,
        DataEnum.TCGA_BRCA.name,
        DataEnum.TCGA_GBM.name,
        DataEnum.AML.name,
        DataEnum.BLCA.name,
        DataEnum.BRCA.name,
        DataEnum.LIHC.name,
        DataEnum.PRAD.name,
        DataEnum.WT.name,
    ]:
        hidden_feats = [i[1] for k, i in dataset.graph.shape.items() if len(i) > 1]
    else:
        hidden_feats = dataset.graph.shape[list(dataset.graph.shape.keys())[0]][1]

    model = MultiGraphGCN(
        stack_types=config["stack_types"],
        hidden_feats=hidden_feats,
        hid_emb=config["hidden_embeedings"],
        reverse_attention=config["reverse_attention"],
        rel_names=dataset.graph.etypes,
        num_patients=dataset.graph.num_patients,
        num_class=dataset.graph.num_class,
        args=args,
        combination=config,
        two_level_attention=config["two_level_attention"],
        device=device,
        omics_shapes=dataset.graph.shape,
    ).to(device)

    optimizer = create_optimizer(args=config, model=model)
    data = {
        omics_train_type: dataset.graph.nodes["patient"].data[omics_train_type]
        for omics_train_type in dataset.graph.etypes
    }

    data = sort_data_order(args=args, train_data=data, forwards=True)
    range_data = np.arange(len(dataset.graph.label))
    alltrain_idx, test_idx = train_test_split(
        range_data,
        test_size=0.2,
        shuffle=True,
        stratify=dataset.graph.label.cpu(),
        random_state=random_state,
    )
    train_idx, val_idx = train_test_split(
        alltrain_idx,
        test_size=0.25,
        shuffle=True,
        stratify=dataset.graph.label[alltrain_idx].cpu(),
        random_state=random_state,
    )
    masking_dict = masking(
        range_data=range_data, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
    )
    criterion = torch.nn.CrossEntropyLoss()
    criterion1_triplet = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=torch.nn.PairwiseDistance()
    )
    model_parameters = {"best_model": None}
    best_f1_macro_val = 0
    early_stopping = 0

    for epoch in tqdm(range(args.epochs)):
        (omics_attention_forward_train, feature_attention_forward_train) = model_train(
            model=model,
            criterion=criterion,
            criterion1_triplet=criterion1_triplet,
            optimizer=optimizer,
            graph=dataset.graph,
            label=dataset.graph.label,
            train_data=data,
            masking_dict=masking_dict,
        )

        f1_macro_val = model_evaluate(
            model=model,
            graph=dataset.graph,
            label=dataset.graph.label,
            train_data=data,
            masking_dict=masking_dict,
        )

        if f1_macro_val > best_f1_macro_val:
            best_f1_macro_val = f1_macro_val
            model_parameters = {"best_model": deepcopy(model.state_dict())}
            fin_omics_attention_forward_train = omics_attention_forward_train
            fin_feature_attention_forward_train = feature_attention_forward_train
            early_stopping = 0
        else:
            early_stopping += 1
        if early_stopping == args.early_stopping:
            break

    model.load_state_dict(model_parameters["best_model"])

    (val_accuracy, f1_test_macro, f1_test_weighted, matthews_corrcoef_test) = (
        model_test(
            model=model,
            graph=dataset.graph,
            label=dataset.graph.label,
            data=data,
            masking_dict=masking_dict,
        )
    )
    return (
        val_accuracy,
        f1_test_macro,
        f1_test_weighted,
        matthews_corrcoef_test,
        fin_omics_attention_forward_train,
        fin_feature_attention_forward_train,
        dataset,
    )


def run_igcn(args, file_path: Path, hyperparameters: Dict):

    combinations = list(product(*hyperparameters.values()))
    for combination in combinations:
        hyper = {
            "similarity_metrix": combination[0],
            "optimizer": combination[1],
            "lr": combination[2],
            "weight_decay": combination[3],
            "stack_types": combination[4],
            "hidden_embeedings": combination[5],
            "reverse_attention": combination[6],
            "aggregator_type": combination[7],
            "dropout": combination[8],
            "two_level_attention": combination[9],
        }
        all_runs_omics = defaultdict()
        all_runs_attention_features_no_perclass = defaultdict()
        all_runs_attention_features_score = defaultdict()
        all_runs = defaultdict(lambda: defaultdict(list))
        for rs in RANDOM_SEEDS:

            dict_key = "_".join([str(i) for i in combination])
            (
                val_accuracy,
                f1_test_macro,
                f1_test_weighted,
                matthews_corrcoef_test,
                omics_attention_forward_train,
                feature_attention_forward_train,
                dataset,
            ) = run_model(config=hyper, args=args, path=file_path, random_state=rs)
            all_runs[dict_key]["val_accuracy"].append(val_accuracy)
            all_runs[dict_key]["f1_test_macro"].append(f1_test_macro)
            all_runs[dict_key]["f1_test_weighted"].append(f1_test_weighted)
            all_runs[dict_key]["matthews_corrcoef_test"].append(matthews_corrcoef_test)

            attention_weights_omics = feature_level_attention(
                weights=feature_attention_forward_train,
                dataset=dataset.graph,
                train_test_val="train_forward",
                attention_types=combination[16],
                per_class_attention=False,
            )

            all_runs_attention_features_no_perclass[
                f"{rs}_feature_attention_forward_train"
            ] = attention_weights_omics
            all_runs_attention_features_score[f"{rs}"] = feature_attention_forward_train

            all_runs_omics[f"{rs}_omics_attention_forward_train"] = (
                omics_attention_forward_train
            )

        mrr_dictionary = defaultdict()
        for omics in TCGA_BRCA:
            mrr_dictionary[omics] = mrr(
                all_runs_attention_features_score,
                omics,
                dataset.graph.features_list[omics],
            )
        with open(f"{file_path}/results/{args.dataset}_mrr.pkl", "wb") as file:
            pickle.dump(mrr_dictionary, file)
        pd.DataFrame(all_runs).to_csv(
            f"{file_path}/results/result_{dict_key}_{args.dataset}_{combination[16]}_perclass_{combination[18]}.csv"
        )
