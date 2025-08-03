import random
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from dgl.heterograph import DGLHeteroGraph
from sklearn.metrics import f1_score, matthews_corrcoef
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss, TripletMarginWithDistanceLoss
from torch.optim import Optimizer

from model import MultiGraphGCN


def calculate_loss(
    label: torch.Tensor, pred: torch.Tensor, criterion: CrossEntropyLoss
) -> torch.Tensor:
    return criterion(pred, label)


def triplet_data(label: torch.Tensor) -> List[Tuple[int, int, int]]:
    triplet_list = []
    for anchor in label:
        anchor = anchor.item()
        anchor_label = label[anchor].item()
        positive_index = random.choice(np.where(label.cpu() == anchor_label)[0])
        negative_index = random.choice(np.where(label.cpu() != anchor_label)[0])
        triplet_list.append((anchor, positive_index, negative_index))
    return triplet_list


def triplet_loss(
    label: torch.Tensor,
    out: torch.Tensor,
    criterion1_triplet: TripletMarginWithDistanceLoss,
    masking_dict: Dict[str, torch.Tensor],
    train_test: str,
) -> torch.Tensor:

    triplet = triplet_data(label)
    triplet = [tri for tri, mask in zip(triplet, masking_dict[train_test]) if mask]
    losses = torch.tensor(0.0).to("cuda:0")
    counts = 0
    for triplet in triplet:
        anchor_emb = out[triplet[0]]
        pos_emb = out[triplet[1]]
        neg_emb = out[triplet[2]]
        loss = criterion1_triplet(anchor_emb, pos_emb, neg_emb)
        losses += loss
        counts += 1
    return losses / counts


def create_optimizer(
    args: Dict[str, Union[str, float, bool, int]], model: torch.nn.Module
) -> Optimizer:
    opt_lower = args["optimizer"].lower()
    parameters_model = list(model.parameters())

    opt_args = dict(lr=args["lr"], weight_decay=args["weight_decay"])

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer_model = optim.Adam(parameters_model, **opt_args)
    elif opt_lower == "adamw":
        optimizer_model = optim.AdamW(parameters_model, **opt_args)
    elif opt_lower == "adadelta":
        optimizer_model = optim.Adadelta(parameters_model, **opt_args)
    elif opt_lower == "radam":
        optimizer_model = optim.RAdam(parameters_model, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        optimizer_model = optim.SGD(parameters_model, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer_model


def masking_node_edges(graph, input_data, node_masking_ratio=0.3):
    input_data_masked = {key: data.clone() for key, data in input_data.items()}
    input_data_masked_dict = defaultdict()
    masked_dict = defaultdict()
    for idx, (key, data) in enumerate(input_data_masked.items()):
        if idx != 0:
            node_masked = torch.bernoulli(
                torch.full(data.shape, 1 - node_masking_ratio)
            ).bool()
            masked_dict[key] = node_masked
            data[~node_masked] = 0.0
            input_data_masked_dict[key] = data
        else:
            input_data_masked_dict[key] = data
            masked_dict[key] = None
    return input_data_masked_dict, masked_dict


def model_train(
    model: MultiGraphGCN,
    criterion: CrossEntropyLoss,
    criterion1_triplet: TripletMarginWithDistanceLoss,
    optimizer: Optimizer,
    graph: DGLHeteroGraph,
    label: torch.Tensor,
    train_data: Dict[str, torch.Tensor],
    masking_dict: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    model.train()
    optimizer.zero_grad()
    embeddings, pred, omics_attention_forward, feature_attention_forward = model(
        graph=graph, input_data=train_data
    )
    label_train = label[masking_dict["train_idx"]]
    pred_train = pred[masking_dict["train_idx"]]
    loss = calculate_loss(label=label_train, pred=pred_train, criterion=criterion)
    additional_loss = triplet_loss(
        label, embeddings, criterion1_triplet, masking_dict, "train_idx"
    )
    loss += additional_loss
    loss.backward()
    optimizer.step()
    accuracy = (pred_train.argmax(dim=1) == label_train).float().mean()
    f1_macro = f1_score(
        pred_train.argmax(dim=1).cpu(), label_train.cpu(), average="macro"
    )
    f1_weighted = f1_score(
        pred_train.argmax(dim=1).cpu(), label_train.cpu(), average="weighted"
    )
    matthews_corrcoef_ = matthews_corrcoef(
        pred_train.argmax(dim=1).cpu(), label_train.cpu()
    )
    print(
        f"Train loss:{loss}, Train accuracy:{accuracy}, f1_macro:{f1_macro}, f1_weighted:{f1_weighted}, matthews_corrcoef_:{matthews_corrcoef_}"
    )
    return omics_attention_forward, feature_attention_forward


@torch.no_grad()
def model_evaluate(
    model: MultiGraphGCN,
    graph: DGLHeteroGraph,
    label: torch.Tensor,
    train_data: Dict[str, torch.Tensor],
    masking_dict: Dict[str, torch.Tensor],
) -> float:

    model.eval()
    _, pred, _, _ = model(graph=graph, input_data=train_data)
    label_val = label[masking_dict["val_idx"]]
    pred_val = pred[masking_dict["val_idx"]].argmax(dim=1)
    accuracy = (pred_val == label_val).float().mean()
    f1_macro = f1_score(pred_val.cpu(), label_val.cpu(), average="macro")
    f1_weighted = f1_score(pred_val.cpu(), label_val.cpu(), average="weighted")
    matthews_corrcoef_ = matthews_corrcoef(pred_val.cpu(), label_val.cpu())
    print(
        f"Val accuracy:{accuracy}, f1_macro:{f1_macro}, f1_weighted:{f1_weighted}, matthews_corrcoef_:{matthews_corrcoef_}"
    )
    return f1_macro


@torch.no_grad()
def model_test(
    model: MultiGraphGCN,
    graph: DGLHeteroGraph,
    label: torch.Tensor,
    data: Dict[str, torch.Tensor],
    masking_dict: Dict[str, torch.Tensor],
) -> Tuple[float, float, float, float]:

    model.eval()
    _, pred, _, _ = model(graph=graph, input_data=data)
    label_test = label[masking_dict["test_idx"]]
    pred_test = pred[masking_dict["test_idx"]].argmax(dim=1)
    val_accuracy = (pred_test == label_test).float().mean()
    f1_test_macro = f1_score(pred_test.cpu(), label_test.cpu(), average="macro")
    f1_test_weighted = f1_score(pred_test.cpu(), label_test.cpu(), average="weighted")
    matthews_corrcoef_test = matthews_corrcoef(pred_test.cpu(), label_test.cpu())
    print(
        f"Test accuracy:{val_accuracy}, f1_test_macro:{f1_test_macro}, f1_test_weighted:{f1_test_weighted}, matthews_corrcoef_test:{matthews_corrcoef_test}"
    )
    return val_accuracy.item(), f1_test_macro, f1_test_weighted, matthews_corrcoef_test
