import os
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
from helper import feature_level_attention
from enum_holder import DataEnum
from helper import masking, mrr, sort_data_order, return_dicitonaries_key
from model import MultiGraphGCN
from model_config import RANDOM_SEEDS
from pre_process_data import MultiOmicsData, create_mrr_dataset
from train_eval import create_optimizer, model_evaluate_1, model_test_1, model_train_1


def run_model(
    config: Dict, args: Any, path: Path, random_state: int
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

    data = sort_data_order(dataset=args.dataset, train_data=data, forwards=True)
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
        dataset=args.dataset,
        range_data=range_data, 
        train_idx=train_idx, 
        val_idx=val_idx, 
        test_idx=test_idx
    )
    criterion = torch.nn.CrossEntropyLoss()
    criterion1_triplet = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=torch.nn.PairwiseDistance()
    )
    model_parameters = {"best_model": None}
    best_f1_macro_val = 0
    early_stopping = 0

    for epoch in tqdm(range(args.epochs)):
        (
        # first_omics_attention, 
        first_feature_attention,
        # first_omics_attention_rev, 
        first_feature_attention_rev,
        # second_omics_attention, 
        # second_feature_attention,
        # second_omics_attention_rev, 
        # second_feature_attention_rev
        ) = model_train_1(
            model=model,
            criterion=criterion,
            criterion1_triplet=criterion1_triplet,
            optimizer=optimizer,
            graph=dataset.graph,
            label=dataset.graph.label,
            train_data=data,
            masking_dict=masking_dict,
            device=device,
            masking_input=config['masking_input'],
            node_masking_ratio=config['node_masking_ratio']
        )

        f1_macro_val = model_evaluate_1(
            model=model,
            graph=dataset.graph,
            label=dataset.graph.label,
            train_data=data,
            masking_dict=masking_dict,
            masking_input=config['masking_input'],
            node_masking_ratio=config['node_masking_ratio']
        )

        if f1_macro_val > best_f1_macro_val:
            best_f1_macro_val = f1_macro_val
            model_parameters = {"best_model": deepcopy(model.state_dict())}          
            # fin_first_omics_attention = first_omics_attention 
            fin_first_feature_attention = first_feature_attention
            # fin_first_omics_attention_rev = first_omics_attention_rev 
            fin_first_feature_attention_rev = first_feature_attention_rev
            # fin_second_omics_attention = second_omics_attention 
            # fin_second_feature_attention = second_feature_attention
            # fin_second_omics_attention_rev = second_omics_attention_rev 
            # fin_second_feature_attention_rev = second_feature_attention_rev
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
    ) = model_test_1(
        model=model,
        graph=dataset.graph,
        label=dataset.graph.label,
        data=data,
        masking_dict=masking_dict,
        masking_input=config['masking_input'],
        node_masking_ratio=config['node_masking_ratio']
    )
    return (
        test_accuracy,
        f1_test_macro,
        f1_test_weighted,
        matthews_corrcoef_test,
        # fin_first_omics_attention,
        fin_first_feature_attention,
        # fin_first_omics_attention_rev,
        fin_first_feature_attention_rev,
        # fin_second_omics_attention,
        # fin_second_feature_attention,
        # fin_second_omics_attention_rev,
        # fin_second_feature_attention_rev,
        dataset,
    )


def run_1(args: Any, file_path: Path, hyperparameters: Dict) -> None:
    if not os.path.exists("results"):
        os.makedirs("results", exist_ok=True)
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
            "masking_input": combination[10],
            "node_masking_ratio": combination[11]
        }
        all_runs_attention_features_no_perclass = defaultdict()
        all_runs_attention_features_perclass = defaultdict()
        
        all_runs = defaultdict(list)
        all_runs_omics = defaultdict()
        all_runs_attention_features_score = defaultdict()
        dict_key = "_".join([str(i) for i in combination])
        for rs in RANDOM_SEEDS:
            # all_runs_attention_features_score = defaultdict()
            (
                test_accuracy,
                f1_test_macro,
                f1_test_weighted,
                matthews_corrcoef_test,
                # fin_first_omics_attention,
                fin_first_feature_attention,
                # fin_first_omics_attention_rev,
                fin_first_feature_attention_rev,
                # fin_second_omics_attention,
                # fin_second_feature_attention,
                # fin_second_omics_attention_rev,
                # fin_second_feature_attention_rev,
                dataset,
            ) = run_model(config=hyper, args=args, path=file_path, random_state=rs)
            all_runs["test_accuracy"].append(test_accuracy)
            all_runs["f1_test_macro"].append(f1_test_macro)
            all_runs["f1_test_weighted"].append(f1_test_weighted)
            all_runs["matthews_corrcoef_test"].append(matthews_corrcoef_test)

            # """Per_class = False"""
            # first_feature_attention_forward = feature_level_attention(
            #     weights=fin_first_feature_attention,
            #     dataset=dataset.graph,
            #     train_test_val="train_forward",
            #     attention_types="all_features",
            #     per_class_attention=False,
            # )

            # all_runs_attention_features_no_perclass[
            #     f"{rs}_first_feature_attention_forward"
            # ] = first_feature_attention_forward
            
            # try:
            #     first_feature_attention_forward_rev = feature_level_attention(
            #         weights=fin_first_feature_attention_rev,
            #         dataset=dataset.graph,
            #         train_test_val="train_reverse",
            #         attention_types="30_top",
            #         per_class_attention=False,
            #     )

            #     all_runs_attention_features_no_perclass[
            #         f"{rs}_first_feature_rev_attention_forward"
            #     ] = first_feature_attention_forward_rev
            # except AttributeError:
            #     pass
            
            # second_feature_attention_forward = feature_level_attention(
            #     weights=fin_second_feature_attention,
            #     dataset=dataset.graph,
            #     train_test_val="train_forward",
            #     attention_types="30_top",
            #     per_class_attention=False,
            # )

            # all_runs_attention_features_no_perclass[
            #     f"{rs}_second_feature_attention_forward"
            # ] = second_feature_attention_forward
            
            
            
            
            # second_feature_attention_forward_rev = feature_level_attention(
            #     weights=fin_second_feature_attention_rev,
            #     dataset=dataset.graph,
            #     train_test_val="train_reverse",
            #     attention_types="30_top",
            #     per_class_attention=False,
            # )

            # all_runs_attention_features_no_perclass[
            #     f"{rs}_second_feature_rev_attention_forward"
            # ] = second_feature_attention_forward_rev



            # """Per_class = True"""
            # first_feature_attention_forward = feature_level_attention(
            #     weights=fin_first_feature_attention,
            #     dataset=dataset.graph,
            #     train_test_val="train_forward",
            #     attention_types="30_top",
            #     per_class_attention=True,
            # )

            # all_runs_attention_features_perclass[
            #     f"{rs}_first_feature_attention_forward"
            # ] = first_feature_attention_forward
            
            # try:
            #     first_feature_attention_forward_rev = feature_level_attention(
            #         weights=fin_first_feature_attention_rev,
            #         dataset=dataset.graph,
            #         train_test_val="train_reverse",
            #         attention_types="30_top",
            #         per_class_attention=True,
            #     )

            #     all_runs_attention_features_perclass[
            #         f"{rs}_first_feature_rev_attention_forward"
            #     ] = first_feature_attention_forward_rev
            # except AttributeError:
            #     pass
            
            # second_feature_attention_forward = feature_level_attention(
            #     weights=fin_second_feature_attention,
            #     dataset=dataset.graph,
            #     train_test_val="train_forward",
            #     attention_types="30_top",
            #     per_class_attention=True,
            # )

            # all_runs_attention_features_perclass[
            #     f"{rs}_second_feature_attention_forward"
            # ] = second_feature_attention_forward
            
            
            
            
            # second_feature_attention_forward_rev = feature_level_attention(
            #     weights=fin_second_feature_attention_rev,
            #     dataset=dataset.graph,
            #     train_test_val="train_reverse",
            #     attention_types="30_top",
            #     per_class_attention=True,
            # )

            # all_runs_attention_features_perclass[
            #     f"{rs}_second_feature_rev_attention_forward"
            # ] = second_feature_attention_forward_rev
            
            # all_runs_attention_features_score[f"fin_first_feature_attention_{rs}"] = fin_first_feature_attention
            # all_runs_attention_features_score[f"fin_first_rev_feature_attention{rs}"] = fin_first_feature_attention_rev
            # all_runs_attention_features_score[f"fin_second_feature_attention_{rs}"] = fin_second_feature_attention
            # all_runs_attention_features_score[f"fin_second_rev_feature_attention{rs}"] = fin_second_feature_attention_rev
            # with open(f"results/{args.dataset}_all_runs_attention_features_score_att_score_fin_final_12_NEW_br_{rs}.pkl", "wb") as file:
            #     pickle.dump(all_runs_attention_features_score, file)
            # all_runs_omics[f"fin_first_omics_attention_{rs}"] = fin_first_omics_attention
            # all_runs_omics[f"fin_first_rev_omics_attention{rs}"] = fin_first_omics_attention_rev
            # all_runs_omics[f"fin_second_omics_attention_{rs}"] = fin_second_omics_attention
            # all_runs_omics[f"fin_second_rev_omics_attention{rs}"] = fin_second_omics_attention_rev
            # with open(f"results/000_{args.dataset}_{dict_key}_fin_first_feature_attention_{rs}.pkl", "wb") as file:
            #     pickle.dump(fin_first_feature_attention, file)
            # with open(f"results/000_{args.dataset}_{dict_key}_fin_first_feature_attention_rev_{rs}.pkl", "wb") as file:
            #     pickle.dump(fin_first_feature_attention_rev, file)
        # with open(f"results/{args.dataset}_{dict_key}_all_runs_attention_features_score_att_score_fin_final.pkl", "wb") as file:
        #     pickle.dump(all_runs_attention_features_score, file)
        # with open(f"results/{args.dataset}_all_runs_omics_fin_final.pkl", "wb") as file:
        #     pickle.dump(all_runs_omics, file)
        # with open(f"results/{args.dataset}_all_runs_attention_features_score_att_score_fin_final_12_NEW_.pkl", "wb") as file:
        #     pickle.dump(all_runs_attention_features_score, file)
        # with open(f"results/{args.dataset}_{dict_key}_all_runs_attention_features_no_perclass_final.pkl", "wb") as file:
        #     pickle.dump(all_runs_attention_features_no_perclass, file)
        # with open(f"results/{args.dataset}_all_runs_attention_features_score_att_score_fin_final.pkl", "wb") as file:
        #     pickle.dump(all_runs_attention_features_score, file)
        # with open(f"results/{args.dataset}_all_runs_attention_features_no_perclass_final.pkl", "wb") as file:
        #     pickle.dump(all_runs_attention_features_no_perclass, file)
        # with open(f"results/{args.dataset}_{dict_key}_all_runs_attention_features_perclass_final.pkl", "wb") as file:
        #     pickle.dump(all_runs_attention_features_perclass, file)

        # with open(f"results/{args.dataset}_all_runs_omics_final.pkl", "wb") as file:
        #     pickle.dump(all_runs_omics, file)
            
            
            
        pd.DataFrame(all_runs).to_csv(f"results/{dict_key}_{args.dataset}_1.csv")
        
        mrr_dictionary = defaultdict(defaultdict)
        list_of_keys = return_dicitonaries_key(all_runs_attention_features_score)
        # for omics in create_mrr_dataset(args.dataset):
        for omics in ['meth', "mirna", "expression"]:
            for keys in list_of_keys:
                # try:
                mrr_dictionary[keys][omics] = mrr(
                    all_runs_attention_features_score=all_runs_attention_features_score,
                    omics=omics,
                    keys=keys,
                    feature_lists=dataset.graph.features_list,
                )
                # except:
                #     pass
        with open(f"results/{args.dataset}_{dict_key}_mrr_fin_1.pkl", "wb") as file:
            pickle.dump(mrr_dictionary, file)
            

    print(
        f'accuracy{np.mean(all_runs["test_accuracy"])}±{np.std(all_runs["test_accuracy"])},\n'
        f'f1_test_macro:{np.mean(all_runs["f1_test_macro"])}±{np.std(all_runs["f1_test_macro"])},\n'
        f'f1_test_weighted:{np.mean(all_runs["f1_test_weighted"])}±{np.std(all_runs["f1_test_weighted"])},\n'
        f'matthews_corrcoef_test:{np.mean(all_runs["matthews_corrcoef_test"])}±{np.std(all_runs["matthews_corrcoef_test"])}'
    )
        
    # all_runs_omics
    
    
    
