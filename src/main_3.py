import argparse
import os
from enum_holder import DataEnum


import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
import pprint
from helper import masking, mrr, sort_data_order, return_dicitonaries_key
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pre_process_data import MultiOmicsData, create_mrr_dataset, get_mask, get_mask_wrapper
from train_eval import create_optimizer, model_evaluate_1, model_test_1, model_train_1, model_train_3
from datetime import datetime
from tqdm import tqdm
import random
from typing import Any, Dict, Tuple
from model import MultiGraphGCN

import torch.optim as optim
from collections import defaultdict
from copy import deepcopy
from itertools import product
import pickle

def run_model(
    config: Dict, args: Any, path: Path,
) -> Tuple[float, torch.Tensor, Dict[str, torch.Tensor], MultiOmicsData]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        DataEnum.BRCA_M.name,
        DataEnum.KIPAN.name,
        DataEnum.LGG.name,
    ]:
        hidden_feats = [i[1] for k, i in dataset.graph.shape.items() if len(i) > 1]
    else:
        hidden_feats = dataset.graph.shape[list(dataset.graph.shape.keys())[0]][1]

    data = {
    omics_train_type: dataset.graph.nodes["patient"].data[omics_train_type]
    for omics_train_type in dataset.graph.etypes
    }
    data = sort_data_order(dataset=args.dataset, train_data=data, forwards=True)
    mask_train = get_mask(3, data["meth"][dataset.graph.idx_dict['tr']].shape[0], config['missing_rate'])
    mask_train = torch.from_numpy(np.asarray(mask_train, dtype=np.float32)).to(device)
    data_train = defaultdict()
    data_clone_tr = {k: v.clone() for k, v in data.items()}
    data_clone_te = {k: v.clone() for k, v in data.items()}
    for idx, (k, v) in enumerate(data_clone_tr.items()):
        # if idx != 0:
        masked_dim_train = torch.unsqueeze(mask_train[:, idx], 1)
        v[dataset.graph.idx_dict['tr']] = v[dataset.graph.idx_dict['tr']] * masked_dim_train
        data_train[k] = v
    mask_train = mask_train
    mask_test = get_mask_wrapper(3, data_clone_te["meth"][dataset.graph.idx_dict['te']].shape[0], config['missing_rate'])
    mask_test = torch.from_numpy(np.asarray(mask_test, dtype=np.float32)).to(device)
    data_test = defaultdict()
    for idx, (k, v) in enumerate(data_clone_te.items()):
        # if idx != 0:
        masked_dim_test = torch.unsqueeze(mask_test[:, idx], 1)
        v[dataset.graph.idx_dict['te']] = v[dataset.graph.idx_dict['te']] * masked_dim_test
        data_test[k] = v
    mask_test = mask_test  

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
    range_data = np.arange(len(dataset.graph.label))
    masking_dict = masking(
    dataset=args.dataset, 
    range_data=range_data, 
    train_idx= dataset.graph.idx_dict['tr'], 
    test_idx=dataset.graph.idx_dict['te']
)
    global_acc = 0.
    if dataset.graph.num_class == 2:
        best_eval = {"acc": None, 
                        "f1":None, 
                        "auc":None}
        
    else:
        best_eval = {"acc": None, 
                        "f1w":None, 
                        "f1m":None}
    print("\nTraining...")
    # optimizer = torch.optim.RAdam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=0.2)
    criterion = torch.nn.CrossEntropyLoss()
    criterion1_triplet = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=torch.nn.PairwiseDistance()
    )
    for epoch in tqdm(range(args.epochs)):
        res1, res2, res3 = model_train_3(config=config, 
            graph=dataset.graph, 
            idx_dict=dataset.graph.idx_dict, 
            model=model, 
            device=device,
            data_train=data_train,
            data=data,
            data_test=data_test,
            label=dataset.graph.label,
            mask_train=mask_train,
            masking_dict=masking_dict,
            optimizer=optimizer,
            criterion=criterion,
            criterion1_triplet=criterion1_triplet,
            epoch=epoch)

        if res1 > global_acc:
            global_acc = res1
            if dataset.graph.num_class == 2:
                best_eval = {"acc": res1, 
                                "f1":res2, 
                                "auc":res3}
            else:
                best_eval = {"acc": res1, 
                                "f1w":res2, 
                                "f1m":res3}

    return best_eval

def run_3(args: Any, file_path: Path, hyperparameters: Dict) -> None:
    if not os.path.exists("results"):
        os.makedirs("results", exist_ok=True)
    combinations = list(product(*hyperparameters.values()))
    saving_path = Path(__file__).parent.parent / "clclsa_datasets" / "results"
    print(saving_path)
    combinations = list(product(*hyperparameters.values()))
    for combination in combinations:
        dict_key = "_".join([str(i) for i in combination])
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
            "masking_input": combination[10],
            "missing_rate": combination[11],
            "step_size": combination[12],
            "test_inverval": combination[13],
        }
        best_eval = run_model(config=hyper, args=args, path=file_path)

        with open(f"{dict_key}.pkl", "wb") as file:
            pickle.dump(best_eval, file)