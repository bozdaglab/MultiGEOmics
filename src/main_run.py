from pathlib import Path
from enum_holder import DataEnum
from main_1 import run_1
from main_2 import run_2
from main_3 import run_3
from model_config import load_parser

args = load_parser()
file_path = Path(__file__).resolve().parent.parent / "dataset"

hyperparameters = {
    "similarity_metrix": ["diff"],
    "optimizer": ["adam"],
    "lr": [0.0001],
    "weight_decay": [1e-2],
    "stack_types": ["stack"],
    "hidden_embeedings": [256],
    "reverse_attention": [False],
    "aggregator_type": ["pool"],
    "dropout": [0.2],
    "two_level_attention": [False],
    "masking_input": [False],
    "missing_rate": [0.1],
    "step_size": [500],
    "test_inverval": [20],
}

if args.dataset in [
    DataEnum.AML.name,
    DataEnum.BLCA.name,
    DataEnum.BRCA.name,
    DataEnum.LIHC.name,
    DataEnum.PRAD.name,
    DataEnum.WT.name,
]:
    run_2(args=args, file_path=file_path/ "data_for_complete_scenario", hyperparameters=hyperparameters)
elif args.dataset in [
    DataEnum.ADNI.name,
    DataEnum.ROSMAP.name,
    DataEnum.TCGA_BRCA.name,
    DataEnum.TCGA_GBM.name,
]:
    run_1(args=args, file_path=file_path/ "data_for_complete_scenario", hyperparameters=hyperparameters)
elif args.dataset in [
    DataEnum.BRCA_M.name,
    DataEnum.KIPAN.name,
    DataEnum.ROSMAP_M.name,
    DataEnum.LGG.name,
]:
    run_3(args=args, file_path=file_path/ "data_for_missing_scenario", hyperparameters=hyperparameters)
else:
    raise ValueError
