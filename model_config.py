import argparse

ROSMAP = []
TCGA_BRCA = ["meth", "mirna", "expression"]
TCGA_GBM = ["mirna", "expression"]
ADNI = []
AML = ["miRNA", "mRNA"]
BLCA = ["snv", "miRNA", "mRNA"]
BRCA = ["snv", "miRNA", "mRNA"]
LIHC = ["snv", "miRNA", "mRNA"]
PRAD = ["snv", "miRNA", "mRNA"]
WT = ["miRNA", "mRNA"]

RANDOM_SEED = 1030
RANDOM_SEEDS = [3, 12, 26, 39, 44, 64, 66, 75, 87, 91]
MASKING = ["train_idx", "val_idx", "test_idx"]
ADNI_ORDER = [["snps", "bile"], ["snps", "lipids"]]


def load_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="AML", help="dataset name")
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train."
    )
    parser.add_argument(
        "--early_stopping", type=int, default=150, help="early stopping."
    )
    return parser.parse_args()
