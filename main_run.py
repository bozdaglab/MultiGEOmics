from pathlib import Path

from enum_holder import DataEnum
from main_DeepKEGG import run_deepkegg
from main_IGCN import run_igcn
from model_config import load_parser

args = load_parser()
file_path = Path(__file__).resolve().parent / "dataset"
"""for normal graphs"""
hyperparameters = {
    "similarity_metrix": ["diff"],
    "optimizer": ["adam"],
    "lr": [0.0001],
    "weight_decay": [1e-2],
    "stack_types": ["stack", "mean", "sum"],
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
    run_deepkegg(args=args, file_path=file_path, hyperparameter=hyperparameters)
elif args.dataset in [
    DataEnum.ADNI.name,
    DataEnum.ROSMAP.name,
    DataEnum.TCGA_BRCA.name,
    DataEnum.TCGA_GBM.name,
    DataEnum.toy.name,
]:
    run_igcn(args=args, file_path=file_path, hyperparameters=hyperparameters)
else:
    raise ValueError
