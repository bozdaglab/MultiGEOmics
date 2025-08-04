from pathlib import Path

from enum_holder import DataEnum
from main_2 import run_2
from main_1 import run_1
from model_config import load_parser

args = load_parser()
file_path = Path(__file__).resolve().parent / "dataset"

hyperparameters = {
    "similarity_metrix": ["diff"],
    "optimizer": ["adam"],
    "lr": [0.0001],
    "weight_decay": [1e-2],
    "stack_types": ["stack"],
    "hidden_embeedings": [256],
    "reverse_attention": [True],
    "aggregator_type": ["pool"],
    "dropout": [0.2],
    "two_level_attention": [True],
}

if args.dataset in [
    DataEnum.AML.name,
    DataEnum.BLCA.name,
    DataEnum.BRCA.name,
    DataEnum.LIHC.name,
    DataEnum.PRAD.name,
    DataEnum.WT.name,
]:
    run_2(args=args, 
        file_path=file_path, 
        hyperparameters=hyperparameters)
elif args.dataset in [
    DataEnum.ADNI.name,
    DataEnum.ROSMAP.name,
    DataEnum.TCGA_BRCA.name,
    DataEnum.TCGA_GBM.name,
]:
    run_1(args=args, 
        file_path=file_path, 
        hyperparameters=hyperparameters)
else:
    raise ValueError
