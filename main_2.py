from collections import defaultdict
from copy import deepcopy
from itertools import product
from model_config import RANDOM_SEED
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from enum_holder import DataEnum
from helper import masking, sort_data_order
from model import MultiGraphGCN
from pre_process_data import MultiOmicsData
from train_eval import create_optimizer, model_test_2, model_train_2
from pathlib import Path
from typing import Any, Dict
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_model(
    config: Dict, args: Any, path: Path, random_state: int
) -> Dict[str, torch.Tensor]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset.islower():
        args.dataset = args.dataset.upper()

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
        hidden_feats = dataset.graph.shape[0][1]

    model = MultiGraphGCN(
        hid_emb=config["hidden_embeedings"],
        stack_types=config["stack_types"],
        hidden_feats=hidden_feats,
        reverse_attention=config["reverse_attention"],
        rel_names=dataset.graph.etypes,
        num_patients=dataset.graph.num_patients,
        num_class=dataset.graph.num_class,
        args=args,
        combination=config,
        two_level_attention=config["two_level_attention"],
        device=device,
        omics_shapes=dataset.graph.omics_shapes,
    ).to(device)

    optimizer = create_optimizer(args=config, model=model)
    data = {
        omics_train_type: dataset.graph.nodes["patient"].data[omics_train_type]
        for omics_train_type in dataset.graph.etypes
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    all_runs = defaultdict(list)
    for train_idx, test_idx in skf.split(data["mRNA"], dataset.graph.label.cpu()):
        data = sort_data_order(args=args, train_data=data, forwards=True)
        range_data = np.arange(len(dataset.graph.label))
        masking_dict = masking(
            range_data=range_data,
            train_idx=train_idx,
            val_idx=torch.tensor([0]),
            test_idx=test_idx,
        )
        del masking_dict["val_idx"]
        criterion = torch.nn.CrossEntropyLoss()
        criterion1_triplet = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=torch.nn.PairwiseDistance()
        )
        model_parameters = {"best_model": None}
        best_f1_macro_val = 0
        early_stopping = 0

        for _ in tqdm(range(args.epochs)):
            (
                f1_macro,
                omics_attention_forward_train,
                feature_attention_forward_train,
            ) = model_train_2(
                model=model,
                criterion=criterion,
                criterion1_triplet=criterion1_triplet,
                optimizer=optimizer,
                graph=dataset.graph,
                label=dataset.graph.label,
                train_data=data,
                masking_dict=masking_dict,
                device=device,
            )

            if f1_macro >= best_f1_macro_val:
                best_f1_macro_val = f1_macro
                model_parameters = {"best_model": deepcopy(model.state_dict())}
                fin_omics_attention_forward_train = omics_attention_forward_train
                fin_feature_attention_forward_train = feature_attention_forward_train
                early_stopping = 0
            else:
                early_stopping += 1
            if early_stopping == args.early_stopping:
                break

        model.load_state_dict(model_parameters["best_model"])

        (
            test_accuracy,
            f1_test_macro,
            f1_test_weighted,
            matthews_corrcoef_test,
            aupr,
            auc_res,
            f1,
            auprc,
            pre,
            rec_res,
        ) = model_test_2(
            model=model,
            graph=dataset.graph,
            label=dataset.graph.label,
            data=data,
            masking_dict=masking_dict,
        )

        all_runs["test_accuracy"].append(test_accuracy)
        all_runs["f1_test_macro"].append(f1_test_macro)
        all_runs["f1_test_weighted"].append(f1_test_weighted)
        all_runs["matthews_corrcoef_test"].append(matthews_corrcoef_test)
        all_runs["aupr"].append(aupr)
        all_runs["auc_res"].append(auc_res)
        all_runs["f1"].append(f1)
        all_runs["auprc"].append(auprc)
        all_runs["pre"].append(pre)
        all_runs["rec_res"].append(rec_res)
    return all_runs


def run_2(args: Any, file_path: Path, hyperparameters: Dict) -> None:
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
        dict_key = "_".join([str(i) for i in hyper.values()])

        random_state = RANDOM_SEED
        all_runs = run_model(
            config=hyper, args=args, path=file_path, random_state=random_state
        )
        pd.DataFrame(all_runs).to_csv(f"results/{dict_key}_{args.dataset}.csv")
    logger.info(
        f'final_accuracy{np.mean(all_runs["val_accuracy"])},\n'
        f'f1_test_macro:{np.mean(all_runs["f1_test_macro"])},\n'
        f'f1_test_weighted:{np.mean(all_runs["f1_test_weighted"])},\n'
        f'matthews_corrcoef_test:{np.mean(all_runs["matthews_corrcoef_test"])}'
    )
