import logging
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import torch
from dgl.heterograph import DGLHeteroGraph
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss, TripletMarginWithDistanceLoss
from torch.optim import Optimizer

from model import MultiGraphGCN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def kl_loss_function(logvar: Dict[str, torch.Tensor], 
                     mu: Dict[str, torch.Tensor]) -> float:
    keys = list(logvar.keys())
    kl_losses = []
    for r, (omics, data) in enumerate(logvar.items()):
        if r == 0:
            kl = -0.5 * torch.mean(1 + data - mu[keys[r]].pow(2) - data.exp())
        else:
            kl = -0.5 * torch.mean(
                1 + data - (mu[keys[r]] - mu[keys[r - 1]]).pow(2) - data.exp()
            )
        kl_losses.append(kl.item())

    return np.mean(kl_losses)


def rec_loss(x_true: Dict[str, torch.Tensor], 
             recons: Dict[str, torch.Tensor], 
             tr_te_idx: Dict[str, List[int]], 
             mask: torch.Tensor) -> float:
    recon_losses = []
    for idx, (omics, data) in enumerate(x_true.items()):
        mask_idx = mask[:, idx]
        d = (1 - mask_idx).bool()
        recon_losses.append(
            torch.mean(
                ((data[tr_te_idx["tr"]][d] - recons[omics][tr_te_idx["tr"]][d])) ** 2
            ).item()
        )

    return np.sum(recon_losses)


def calculate_loss(
    label: torch.Tensor, pred: torch.Tensor, criterion: CrossEntropyLoss
) -> torch.Tensor:
    return criterion(pred, label)


def triplet_data(label: torch.Tensor) -> List[Tuple[int]]:
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
    device: torch.device,
) -> torch.Tensor:

    triplet = triplet_data(label)
    triplet = [tri for tri, mask in zip(triplet, masking_dict[train_test]) if mask]
    losses = torch.tensor(0.0).to(device)
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


def masking_node_edges(graph, input_data, node_masking_ratio=0.1):
    input_data_masked = {key: data.clone() for key, data in input_data.items()}
    input_data_masked_dict = defaultdict()
    masked_dict = defaultdict()
    for idx, (key, data) in enumerate(input_data_masked.items()):
        node_masked = torch.bernoulli(
            torch.full(data.shape, 1 - node_masking_ratio)
        ).bool()
        masked_dict[key] = node_masked
        data[~node_masked] = 0.0
        input_data_masked_dict[key] = data
    return input_data_masked_dict, masked_dict


def model_train_1(
    model: MultiGraphGCN,
    criterion: CrossEntropyLoss,
    criterion1_triplet: TripletMarginWithDistanceLoss,
    optimizer: Optimizer,
    graph: DGLHeteroGraph,
    label: torch.Tensor,
    train_data: Dict[str, torch.Tensor],
    masking_dict: Dict[str, torch.Tensor],
    device: torch.device,
    masking_input: bool,
    node_masking_ratio: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    model.train()
    optimizer.zero_grad()
    if masking_input:
        train_data, masked_dict = masking_node_edges(
            graph=graph, input_data=train_data, node_masking_ratio=node_masking_ratio
        )
    (
        embeddings,
        pred,
        first_feature_attention,
        first_feature_attention_rev,
    ) = model(graph=graph, input_data=train_data)
    label_train = label[masking_dict["train_idx"]]
    pred_train = pred[masking_dict["train_idx"]]
    loss = calculate_loss(label=label_train, pred=pred_train, criterion=criterion)
    additional_loss = triplet_loss(
        label=label,
        out=embeddings,
        criterion1_triplet=criterion1_triplet,
        masking_dict=masking_dict,
        train_test="train_idx",
        device=device,
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
    logger.info(
        f"Train loss:{loss:.4f}, Train accuracy:{accuracy:.4f}, f1_macro:{f1_macro:.4f}, f1_weighted:{f1_weighted:.4f}, matthews_corrcoef_:{matthews_corrcoef_:.4f}"
    )
    return (
        first_feature_attention,
        first_feature_attention_rev,
    )


@torch.no_grad()
def model_evaluate_1(
    model: MultiGraphGCN,
    graph: DGLHeteroGraph,
    label: torch.Tensor,
    train_data: Dict[str, torch.Tensor],
    masking_dict: Dict[str, torch.Tensor],
    masking_input: bool,
    node_masking_ratio: float,
) -> float:

    model.eval()
    if masking_input:
        train_data, masked_dict = masking_node_edges(
            graph=graph, input_data=train_data, node_masking_ratio=node_masking_ratio
        )
    _, pred, _, _ = model(graph=graph, input_data=train_data)
    label_val = label[masking_dict["val_idx"]]
    pred_val = pred[masking_dict["val_idx"]].argmax(dim=1)
    accuracy = (pred_val == label_val).float().mean()
    f1_macro = f1_score(pred_val.cpu(), label_val.cpu(), average="macro")
    f1_weighted = f1_score(pred_val.cpu(), label_val.cpu(), average="weighted")
    matthews_corrcoef_ = matthews_corrcoef(pred_val.cpu(), label_val.cpu())
    logger.info(
        f"Val accuracy:{accuracy:.4f}, f1_macro:{f1_macro:.4f}, f1_weighted:{f1_weighted:.4f}, matthews_corrcoef_:{matthews_corrcoef_:.4f}"
    )
    return f1_macro


@torch.no_grad()
def model_test_1(
    model: MultiGraphGCN,
    graph: DGLHeteroGraph,
    label: torch.Tensor,
    data: Dict[str, torch.Tensor],
    masking_dict: Dict[str, torch.Tensor],
    masking_input: bool,
    node_masking_ratio: float,
) -> Tuple[float]:

    model.eval()
    if masking_input:
        data, masked_dict = masking_node_edges(
            graph=graph, input_data=data, node_masking_ratio=node_masking_ratio
        )
    _, pred, _, _ = model(graph=graph, input_data=data)
    label_test = label[masking_dict["test_idx"]]
    pred_test = pred[masking_dict["test_idx"]].argmax(dim=1)
    text_accuracy = (pred_test == label_test).float().mean()
    f1_test_macro = f1_score(pred_test.cpu(), label_test.cpu(), average="macro")
    f1_test_weighted = f1_score(pred_test.cpu(), label_test.cpu(), average="weighted")
    matthews_corrcoef_test = matthews_corrcoef(pred_test.cpu(), label_test.cpu())
    logger.info(
        f"Test accuracy:{text_accuracy:.4f}, f1_test_macro:{f1_test_macro:.4f}, f1_test_weighted:{f1_test_weighted:.4f}, matthews_corrcoef_test:{matthews_corrcoef_test:.4f}"
    )
    return text_accuracy.item(), f1_test_macro, f1_test_weighted, matthews_corrcoef_test


def model_train_2(
    model: MultiGraphGCN,
    criterion: CrossEntropyLoss,
    criterion1_triplet: TripletMarginWithDistanceLoss,
    optimizer: Optimizer,
    graph: DGLHeteroGraph,
    label: torch.Tensor,
    train_data: Dict[str, torch.Tensor],
    masking_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad()
    (
        embeddings,
        pred,
        first_feature_attention,
        first_feature_attention_rev,
    ) = model(graph=graph, input_data=train_data)
    label_train = label[masking_dict["train_idx"]]
    pred_train = pred[masking_dict["train_idx"]]
    loss = calculate_loss(label=label_train, pred=pred_train, criterion=criterion)
    additional_loss = triplet_loss(
        label=label,
        out=embeddings,
        criterion1_triplet=criterion1_triplet,
        masking_dict=masking_dict,
        train_test="train_idx",
        device=device,
    )
    loss += additional_loss
    loss.backward()
    optimizer.step()
    accuracy = (pred_train.argmax(dim=1) == label_train).float().mean()
    f1_macro = f1_score(
        pred_train.argmax(dim=1).cpu(), label_train.cpu(), average="binary"
    )
    f1_weighted = f1_score(
        pred_train.argmax(dim=1).cpu(), label_train.cpu(), average="weighted"
    )
    matthews_corrcoef_ = matthews_corrcoef(
        pred_train.argmax(dim=1).cpu(), label_train.cpu()
    )
    aupr = average_precision_score(label_train.cpu(), pred_train.argmax(dim=1).cpu())
    fpr, tpr, _ = roc_curve(
        label_train.cpu(), pred_train.argmax(dim=1).cpu(), pos_label=1
    )
    auc_res = auc(fpr, tpr)
    pre, rec, _ = precision_recall_curve(
        label_train.cpu(), pred_train.argmax(dim=1).cpu()
    )
    auprc = auc(rec, pre)
    pre_res = precision_score(label_train.cpu(), pred_train.argmax(dim=1).cpu())
    f1 = f1_score(label_train.cpu(), pred_train.argmax(dim=1).cpu())
    logger.info(
        f"Train loss:{loss:.4f}, Train accuracy:{accuracy:.4f}, f1_macro:{f1_macro:.4f}, f1_weighted:{f1_weighted:.4f},\n"
        f"matthews_corrcoef_:{matthews_corrcoef_:.4f}, aupr:{aupr:.4f}, auc:{auc_res:.4f}, f1:{f1:.4f}, auprc:{auprc:.4f}, pre:{pre_res:.4f}"
    )
    return (
        f1_macro,
        first_feature_attention,
        first_feature_attention_rev,
    )


@torch.no_grad()
def model_evaluate_2(
    model: MultiGraphGCN,
    graph: DGLHeteroGraph,
    label: torch.Tensor,
    train_data: Dict[str, torch.Tensor],
    masking_dict: Dict[str, torch.Tensor],
) -> Tuple[float, torch.Tensor, Dict[str, torch.Tensor]]:
    model.eval()
    _, pred, first_feature_attention, first_feature_attention_rev = model(
        graph=graph, input_data=train_data
    )
    label_val = label[masking_dict["val_idx"]]
    pred_val = pred[masking_dict["val_idx"]].argmax(dim=1)
    accuracy = (pred_val == label_val).float().mean()
    f1_macro = f1_score(pred_val.cpu(), label_val.cpu(), average="binary")
    f1_weighted = f1_score(pred_val.cpu(), label_val.cpu(), average="weighted")
    matthews_corrcoef_ = matthews_corrcoef(pred_val.cpu(), label_val.cpu())
    aupr = average_precision_score(label_val.cpu(), pred_val.cpu())
    fpr, tpr, _ = roc_curve(label_val.cpu(), pred_val.cpu(), pos_label=1)
    auc_res = auc(fpr, tpr)
    pre, rec, _ = precision_recall_curve(label_val.cpu(), pred_val.cpu())
    auprc = auc(rec, pre)
    f1 = f1_score(label_val.cpu(), pred_val.cpu())
    pre_res = precision_score(label_val.cpu(), pred_val.cpu())
    logger.info(
        f"Val accuracy:{accuracy:.4f}, f1_macro:{f1_macro:.4f}, f1_weighted:{f1_weighted:.4f}, matthews_corrcoef_:{matthews_corrcoef_:.4f},\n"
        f"aupr:{aupr:.4f}, auc:{auc_res:.4f}, f1:{f1:.4f}, auprc:{auprc:.4f}, pre:{pre_res:.4f}"
    )
    return f1_macro, first_feature_attention, first_feature_attention_rev


@torch.no_grad()
def model_test_2(
    model: MultiGraphGCN,
    graph: DGLHeteroGraph,
    label: torch.Tensor,
    data: Dict[str, torch.Tensor],
    masking_dict: Dict[str, torch.Tensor],
    transfer_learning: bool,
) -> Tuple[float]:
    model.eval()
    (
        embeddings,
        pred,
        first_feature_attention,
        first_feature_attention_rev,
    ) = model(graph=graph, input_data=data)
    if transfer_learning:
        label_test = label
        pred_logits_test = pred
    else:
        test_idx = masking_dict["test_idx"]
        label_test = label[test_idx]
        pred_logits_test = pred[test_idx]
    pred_test = pred_logits_test.argmax(dim=1)
    text_accuracy = (pred_test == label_test).float().mean()
    f1_test_macro = f1_score(label_test.cpu(), pred_test.cpu(), average="binary")
    f1_test_weighted = f1_score(label_test.cpu(), pred_test.cpu(), average="weighted")
    f1 = f1_score(label_test.cpu(), pred_test.cpu())
    pre_res = precision_score(label_test.cpu(), pred_test.cpu())
    rec_res = recall_score(label_test.cpu(), pred_test.cpu())
    matthews_corrcoef_test = matthews_corrcoef(label_test.cpu(), pred_test.cpu())
    probs_class1 = pred_logits_test[:, 1].cpu()
    fpr, tpr, _ = roc_curve(label_test.cpu(), probs_class1)
    auc_res = auc(fpr, tpr)
    pre_curve, rec_curve, _ = precision_recall_curve(label_test.cpu(), probs_class1)
    auprc = auc(rec_curve, pre_curve)
    aupr = average_precision_score(label_test.cpu(), probs_class1)

    logger.info(
        f"Test accuracy:{text_accuracy:.4f}, f1_test_macro:{f1_test_macro:.4f}, f1_test_weighted:{f1_test_weighted:.4f}, "
        f"matthews_corrcoef_test:{matthews_corrcoef_test:.4f}, aupr:{aupr:.4f}, auc:{auc_res:.4f}, f1:{f1:.4f}, "
        f"auprc:{auprc:.4f}, pre:{pre_res:.4f}, rec:{rec_res:.4f}"
    )

    return (
        text_accuracy.item(),
        f1_test_macro,
        f1_test_weighted,
        matthews_corrcoef_test,
        aupr,
        auc_res,
        f1,
        auprc,
        pre_res,
        rec_res,
    )


def model_train_3(
    config: Any,
    graph: DGLHeteroGraph,
    idx_dict: Dict[str, int],
    model: MultiGraphGCN,
    device: torch.device,
    data_train: Dict[str, torch.Tensor],
    data: Dict[str, torch.Tensor],
    data_test: Dict[str, torch.Tensor],
    label: torch.Tensor,
    mask_train: torch.Tensor,
    optimizer: optim,
    masking_dict: Dict[str, torch.Tensor],
    criterion: torch.nn.modules.loss,
    criterion1_triplet: torch.nn.modules.loss,
    epoch: int,
) -> Tuple[Union[float, int]]:

    model.train()
    optimizer.zero_grad()

    (
        embeddings,
        pred,
        final_embeddings,
        first_feature_attention,
        first_feature_attention_rev,
        mu,
        var,
        recons,
    ) = model(graph=graph, input_data=data_train)
    kl_losses = kl_loss_function(var, mu)
    imputation_losses = rec_loss(data, recons, idx_dict, mask_train)
    label_train = label[masking_dict["train_idx"]]
    pred_train = pred[masking_dict["train_idx"]]
    loss = calculate_loss(label=label_train, pred=pred_train, criterion=criterion)
    additional_loss = triplet_loss(
        label=label_train,
        out=embeddings,
        criterion1_triplet=criterion1_triplet,
        masking_dict=masking_dict,
        train_test="train_idx",
        device=device,
    )
    los = kl_losses + additional_loss + imputation_losses + loss
    los.backward()
    optimizer.step()

    if epoch % config["test_inverval"] == 0:
        te_prob = model_test_3(model, data_test, graph)
        label_test = graph.label[masking_dict["test_idx"]]
        pred_test = te_prob[masking_dict["test_idx"]].argmax(dim=1)
        print("\nTest: Epoch {:d}".format(epoch))
        if graph.num_class == 2:
            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1 = f1_score(label_test.cpu(), pred_test.cpu())
            auc = roc_auc_score(label_test.cpu(), pred_test.cpu())
            print(f"Test ACC: {acc:.5f}, F1: {f1:.5f}, AUC: {auc:.5f}")
            return acc, f1, auc
        else:
            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1w = f1_score(label_test.cpu(), pred_test.cpu(), average="weighted")
            f1m = f1_score(label_test.cpu(), pred_test.cpu(), average="macro")
            print(f"Test ACC: {acc:.5f}, F1 weighted : {f1w:.5f}, F1 macro: {f1m:.5f}")
            return acc, f1w, f1m
    else:
        return 0, 0, 0


def model_test_3(model: MultiGraphGCN, 
                 data_test: Dict[str, torch.Tensor], 
                 graph: DGLHeteroGraph) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        _, pred, _, _, _, _, _, _ = model(graph=graph, input_data=data_test)
    return pred
