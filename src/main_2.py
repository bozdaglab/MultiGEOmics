import logging
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from helper import feature_level_attention
from helper import masking, mrr, sort_data_order, return_dicitonaries_key
from typing import Any, Dict
import random
import numpy as np
import pandas as pd
import torch
import pickle
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from enum_holder import DataEnum
from helper import masking, sort_data_order, prepare_new_data
from model import MultiGraphGCN
from model_config import RANDOM_SEED
from pre_process_data import MultiOmicsData, create_mrr_dataset
from train_eval import create_optimizer, model_test_2, model_train_2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_model(
    config: Dict, args: Any, path: Path, random_state: int, transfer_learning: bool
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
        DataEnum.ROSMAP.name,
        DataEnum.TCGA_BRCA.name,
    ]:
        hidden_feats = dataset.graph.shape[0][1]
    else:
        hidden_feats = [i[1] for k, i in dataset.graph.shape.items() if len(i) > 1]

    data = {
            omics_train_type: dataset.graph.nodes["patient"].data[omics_train_type]
            for omics_train_type in dataset.graph.etypes
        }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    all_runs = defaultdict(list)  

    # if transfer_learning:
    #     dataset_list = [DataEnum.BLCA.name, 
    #                         DataEnum.BRCA.name, 
    #                         DataEnum.LIHC.name,
    #                         DataEnum.PRAD.name
    #                         ]
    #     if args.dataset in dataset_list:
    #         dataset_list.remove(args.dataset)
    
        #     input_data = DataEnum.LIHC.name#random.choice(dataset_list)
        # elif args.dataset == DataEnum.AML.name:
        #     input_data = DataEnum.WT.name
        # elif args.dataset == DataEnum.WT.name:
        #     input_data = DataEnum.AML.name  
    transfer_learning_result = defaultdict(defaultdict)
    transfer_learning_result_test = defaultdict()
    all_runs_attention_features_no_perclass = defaultdict()
    all_runs_attention_features_perclass = defaultdict()
    all_runs_attention_features_score = defaultdict()
    all_runs = defaultdict(list)
    all_runs_omics = defaultdict()
    for rs, (train_idx, test_idx) in enumerate(skf.split(data["mRNA"], dataset.graph.label.cpu())):
    # """remove"""
    # for iterator in range(5):
        # """remove"""
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

        data = sort_data_order(dataset=args.dataset, train_data=data, forwards=True)
        range_data = np.arange(len(dataset.graph.label))
        # """remove"""
        # test_idx = np.array(dataset.graph.patient_ids[iterator])
        # train_idx = np.setdiff1d(range_data, test_idx)
        # """remove"""
        masking_dict = masking(
            dataset=args.dataset,
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
            (f1_macro,
        # first_omics_attention,
        first_feature_attention,
        # first_omics_attention_rev, 
        first_feature_attention_rev,
        # second_omics_attention, 
        # second_feature_attention,
        # second_omics_attention_rev, 
        # second_feature_attention_rev 
        )= model_train_2(
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
        # if transfer_learning:
        #     new_dataset = MultiOmicsData(
        #         path=path,
        #         folder_name=input_data,
        #         file_name=f"{input_data}_data",
        #         force_reload=True,
        #         similarity_metrix=config["similarity_metrix"],
        #         device=device,
        #     )
            
        #     new_data = prepare_new_data(new_dataset=new_dataset, dataset=dataset)
        #     new_data = sort_data_order(dataset=input_data, train_data=new_data, forwards=True)
            
        #     (
        #         test_accuracy,
        #         f1_test_macro,
        #         f1_test_weighted,
        #         matthews_corrcoef_test,
        #         aupr,
        #         auc_res,
        #         f1,
        #         auprc,
        #         pre,
        #         rec_res,
        #     ) = model_test_2(
        #         model=model,
        #         graph=new_dataset.graph,
        #         label=new_dataset.graph.label,
        #         data=new_data,
        #         masking_dict=masking_dict,
        #         transfer_learning=transfer_learning
        #     )
        # else:
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
            transfer_learning=transfer_learning
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
        
        """"""
    

        # """Per_class = False"""
        first_feature_attention_forward = feature_level_attention(
            weights=fin_first_feature_attention,
            dataset=dataset.graph,
            train_test_val="train_forward",
            attention_types="all_features",
            per_class_attention=False,
        )

        all_runs_attention_features_no_perclass[
            f"{rs}_first_feature_attention_forward"
        ] = first_feature_attention_forward
        
        """reverse"""
        # first_feature_attention_forward_rev = feature_level_attention(
        #     weights=fin_first_feature_attention_rev,
        #     dataset=dataset.graph,
        #     train_test_val="train_reverse",
        #     attention_types="all_features",
        #     per_class_attention=False,
        # )

        # all_runs_attention_features_no_perclass[
        #     f"{rs}_first_feature_rev_attention_forward"
        # ] = first_feature_attention_forward_rev
        
        
        # second_feature_attention_forward = feature_level_attention(
        #     weights=fin_second_feature_attention,
        #     dataset=dataset.graph,
        #     train_test_val="train_forward",
        #     attention_types="all_features",
        #     per_class_attention=False,
        # )

        # all_runs_attention_features_no_perclass[
        #     f"{rs}_second_feature_attention_forward"
        # ] = second_feature_attention_forward
        
        
        
        
        # second_feature_attention_forward_rev = feature_level_attention(
        #     weights=fin_second_feature_attention_rev,
        #     dataset=dataset.graph,
        #     train_test_val="train_reverse",
        #     attention_types="all_features",
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
        #     attention_types="all_features",
        #     per_class_attention=True,
        # )

        # all_runs_attention_features_perclass[
        #     f"{rs}_first_feature_attention_forward"
        # ] = first_feature_attention_forward
        
        # first_feature_attention_forward_rev = feature_level_attention(
        #     weights=fin_first_feature_attention_rev,
        #     dataset=dataset.graph,
        #     train_test_val="train_reverse",
        #     attention_types="all_features",
        #     per_class_attention=True,
        # )

        # all_runs_attention_features_perclass[
        #     f"{rs}_first_feature_rev_attention_forward"
        # ] = first_feature_attention_forward_rev
        
        
        # second_feature_attention_forward = feature_level_attention(
        #     weights=fin_second_feature_attention,
        #     dataset=dataset.graph,
        #     train_test_val="train_forward",
        #     attention_types="all_features",
        #     per_class_attention=True,
        # )

        # all_runs_attention_features_perclass[
        #     f"{rs}_second_feature_attention_forward"
        # ] = second_feature_attention_forward
        
        
        
        
        # second_feature_attention_forward_rev = feature_level_attention(
        #     weights=fin_second_feature_attention_rev,
        #     dataset=dataset.graph,
        #     train_test_val="train_reverse",
        #     attention_types="all_features",
        #     per_class_attention=True,
        # )

        # all_runs_attention_features_perclass[
        #     f"{rs}_second_feature_rev_attention_forward"
        # ] = second_feature_attention_forward_rev
        
        all_runs_attention_features_score[f"fin_first_feature_attention_{rs}"] = fin_first_feature_attention
        # all_runs_attention_features_score[f"fin_first_rev_feature_attention{rs}"] = fin_first_feature_attention_rev
        # all_runs_attention_features_score[f"fin_second_feature_attention_{rs}"] = fin_second_feature_attention
        # all_runs_attention_features_score[f"fin_second_rev_feature_attention{rs}"] = fin_second_feature_attention_rev

        # all_runs_omics[f"fin_first_omics_attention_{rs}"] = fin_first_omics_attention
        # all_runs_omics[f"fin_first_rev_omics_attention{rs}"] = fin_first_omics_attention_rev
        # all_runs_omics[f"fin_second_omics_attention_{rs}"] = fin_second_omics_attention
        # all_runs_omics[f"fin_second_rev_omics_attention{rs}"] = fin_second_omics_attention_rev

    logger.info(
        f'accuracy{np.mean(all_runs["test_accuracy"])}±{np.std(all_runs["test_accuracy"])},\n'
        f'f1_test_macro:{np.mean(all_runs["f1_test_macro"])}±{np.std(all_runs["f1_test_macro"])},\n'
        f'f1_test_weighted:{np.mean(all_runs["f1_test_weighted"])}±{np.std(all_runs["f1_test_weighted"])},\n'
        f'matthews_corrcoef_test:{np.mean(all_runs["matthews_corrcoef_test"])}±{np.std(all_runs["matthews_corrcoef_test"])}, \n'
        f'AUPRC{np.mean(all_runs["auprc"])}±{np.std(all_runs["auprc"])}, \n'
        f'AUC{np.mean(all_runs["auc_res"])}±{np.std(all_runs["auc_res"])}, \n'
        f'pre{np.mean(all_runs["pre"])}±{np.std(all_runs["pre"])}'
    )        
    with open(f"results/{args.dataset}_all_runs_attention_features_score_att_score_fin_final.pkl", "wb") as file:
        pickle.dump(all_runs_attention_features_score, file)
    with open(f"results/{args.dataset}_all_runs_attention_features_no_perclass_final.pkl", "wb") as file:
        pickle.dump(all_runs_attention_features_no_perclass, file)
        
    # with open(f"results/{args.dataset}_all_runs_attention_features_perclass_final.pkl", "wb") as file:
    #     pickle.dump(all_runs_attention_features_perclass, file)

    # with open(f"results/{args.dataset}_all_runs_omics_final.pkl", "wb") as file:
    #     pickle.dump(all_runs_omics, file)
    mrr_dictionary = defaultdict(defaultdict)
    list_of_keys = return_dicitonaries_key(all_runs_attention_features_score)
    for omics in ['snv', 'miRNA', 'mRNA']:
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
    with open(f"results/{args.dataset}_mrr_fin.pkl", "wb") as file:
        pickle.dump(mrr_dictionary, file)
        """"""
    transfer_data = [args.dataset]
    if transfer_learning:
        dataset_list = [
            DataEnum.BLCA.name, 
            DataEnum.BRCA.name, 
            DataEnum.LIHC.name,
            DataEnum.PRAD.name
            ]
        for ext_data in dataset_list:
            if ext_data not in  transfer_data:
                new_dataset = MultiOmicsData(
                    path=path,
                    folder_name=ext_data,
                    file_name=f"{ext_data}_data",
                    force_reload=True,
                    similarity_metrix=config["similarity_metrix"],
                    device=device,
                )
                
                new_data = prepare_new_data(new_dataset=new_dataset, dataset=dataset)
                new_data = sort_data_order(dataset=ext_data, train_data=new_data, forwards=True)
                
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
                    graph=new_dataset.graph,
                    label=new_dataset.graph.label,
                    data=new_data,
                    masking_dict=masking_dict,
                    transfer_learning=transfer_learning
                )
                transfer_learning_result_test[ext_data] = (
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
                )
                for idd, (train_idx, test_idx) in enumerate(skf.split(new_data["mRNA"], new_dataset.graph.label.cpu())):
                    optimizer = create_optimizer(args=config, model=model)
                    data = sort_data_order(dataset=ext_data, train_data=new_data, forwards=True)
                    range_data = np.arange(len(new_dataset.graph.label))
                    masking_dict = masking(
                        dataset=ext_data,
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
                        f1_macro = model_train_2(
                            model=model,
                            criterion=criterion,
                            criterion1_triplet=criterion1_triplet,
                            optimizer=optimizer,
                            graph=new_dataset.graph,
                            label=new_dataset.graph.label,
                            train_data=new_data,
                            masking_dict=masking_dict,
                            device=device,
                        )

                        if f1_macro >= best_f1_macro_val:
                            best_f1_macro_val = f1_macro
                            model_parameters = {"best_model": deepcopy(model.state_dict())}
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
                        graph=new_dataset.graph,
                        label=new_dataset.graph.label,
                        data=new_data,
                        masking_dict=masking_dict,
                        transfer_learning=transfer_learning
                    )
                    transfer_learning_result[f"{ext_data}_test_accuracy_{idd}"] = test_accuracy
                    transfer_learning_result[f"{ext_data}_f1_test_macro_{idd}"] = f1_test_macro
                    transfer_learning_result[f"{ext_data}_f1_test_weighted_{idd}"] = f1_test_weighted
                    transfer_learning_result[f"{ext_data}_matthews_corrcoef_test_{idd}"] = matthews_corrcoef_test
                    transfer_learning_result[f"{ext_data}_aupr_{idd}"] = aupr
                    transfer_learning_result[f"{ext_data}_auc_res_{idd}"] = auc_res
                    transfer_learning_result[f"{ext_data}_f1_{idd}"] = f1
                    transfer_learning_result[f"{ext_data}_auprc_{idd}"] = auprc
                    transfer_learning_result[f"{ext_data}_pre_{idd}"] = pre
                    transfer_learning_result[f"{ext_data}_rec_res_{idd}"] = rec_res
                # dataset = new_dataset
                transfer_data.append(ext_data)

    return (all_runs, 
            transfer_learning_result_test, 
            transfer_learning_result
    )


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
        (all_runs, 
        transfer_learning_result_test, 
        transfer_learning_result) = run_model(
            config=hyper, args=args, path=file_path, random_state=random_state, transfer_learning=False
        )
        pd.DataFrame(all_runs).to_csv(f"results/{dict_key}_{args.dataset}.csv")
        # with open(f"results/{dict_key}_{args.dataset}_transfer_learning_result_test.pkl", "wb") as file:
        #     pickle.dump(transfer_learning_result_test, file)
       
        # with open(f"results/{dict_key}_{args.dataset}_transfer_learning_result.pkl", "wb") as file:
        #     pickle.dump(transfer_learning_result, file)
       
        
        
            
        logger.info(
            f'accuracy{np.mean(all_runs["test_accuracy"])}±{np.std(all_runs["test_accuracy"])},\n'
            f'f1_test_macro:{np.mean(all_runs["f1_test_macro"])}±{np.std(all_runs["f1_test_macro"])},\n'
            f'f1_test_weighted:{np.mean(all_runs["f1_test_weighted"])}±{np.std(all_runs["f1_test_weighted"])},\n'
            f'matthews_corrcoef_test:{np.mean(all_runs["matthews_corrcoef_test"])}±{np.std(all_runs["matthews_corrcoef_test"])}, \n'
            f'AUPRC{np.mean(all_runs["auprc"])}±{np.std(all_runs["auprc"])}, \n'
            f'AUC{np.mean(all_runs["auc_res"])}±{np.std(all_runs["auc_res"])}, \n'
            f'pre{np.mean(all_runs["pre"])}±{np.std(all_runs["pre"])}'
        )
        all_runs
