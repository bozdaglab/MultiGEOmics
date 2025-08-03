import argparse

ROSMAP = ["meth", "mirna", "expression"]
TCGA_BRCA = ["meth", "mirna", "expression"]
TCGA_GBM = ["mirna", "expression"]
ADNI = ["snps", "bile", "snps", "lipids"]
AML = ["miRNA", "mRNA"]
BLCA = ["snv", "miRNA", "mRNA"]
BRCA = ["snv", "miRNA", "mRNA"]
LIHC = ["snv", "miRNA", "mRNA"]
PRAD = ["snv", "miRNA", "mRNA"]
WT = ["miRNA", "mRNA"]

RANDOM_SEEDS = [3, 12, 26, 39, 44, 64, 66, 75, 87, 91]
MASKING = ["train_idx", "val_idx", "test_idx"]


def load_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="TCGA_BRCA", help="dataset name")
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of epochs to train."
    )
    parser.add_argument(
        "--early_stopping", type=int, default=150, help="early stopping."
    )
    parser.add_argument("--number_runs", type=int, default=10, help="number of runs.")
    return parser.parse_args()
