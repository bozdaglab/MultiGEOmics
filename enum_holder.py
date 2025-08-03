from enum import Enum, auto


class DataEnum(Enum):
    ADNI = auto()
    AML = auto()
    BLCA = auto()
    BRCA = auto()
    LIHC = auto()
    PRAD = auto()
    ROSMAP = auto()
    TCGA_BRCA = auto()
    TCGA_GBM = auto()
    toy = auto()
    WT = auto()


class SimilarityEnum(Enum):
    cosine = auto()
    euclidean = auto()
    coeff = auto()
    spearmanr = auto()
    knn = auto()
    knngraph = auto()
    diff = auto()
