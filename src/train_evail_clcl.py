import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pprint
from helper import masking, mrr, sort_data_order, return_dicitonaries_key

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datetime import datetime
from tqdm import tqdm
import random
import torch.optim as optim
from collections import defaultdict



def model_train_3(params, 
          graph, 
          idx_dict, 
          model, 
          device,
          data_train,
          data,
          data_test,
          labels_trte,
          trte_idx,
          mask_train,
          optimizer):

    exp_name = os.path.join('./exp', f"{params['dataset']}_{datetime.utcnow().strftime('%B_%d_%Y_%Hh%Mm%Ss')}")
    os.makedirs(exp_name, exist_ok=True)
    with open(os.path.join(exp_name, 'config.json'), 'w') as fp:
        json.dump(params, fp, indent=4)
    
    model.train()
    optimizer.zero_grad()
            
    (embeddings, 
    pred, 
    final_embeddings,
    #  first_omics_attention, 
    first_feature_attention,
    # first_omics_attention_rev, 
    first_feature_attention_rev,
    # second_omics_attention, 
    second_feature_attention,
    # second_omics_attention_rev, 
    second_feature_attention_rev,
    mu, var, recons
    ) = model(
        graph=graph, input_data=data_train
    )
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

    if epoch % params['test_inverval'] == 0:
        te_prob = model_test_3(model, data_test, graph)
        label_test = graph.label[masking_dict["test_idx"]]
        pred_test = te_prob[masking_dict["test_idx"]].argmax(dim=1)
        print("\nTest: Epoch {:d}".format(epoch))
        if graph.num_class == 2:
            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1 = f1_score(label_test.cpu(), pred_test.cpu())
            auc = roc_auc_score(label_test.cpu(), pred_test.cpu())
            print(f"Test ACC: {acc:.5f}, F1: {f1:.5f}, AUC: {auc:.5f}")
            if acc > global_acc:
                global_acc = acc
                best_eval = {"acc": acc, 
                                "f1":f1, 
                                "auc":auc}
                
        else:
            acc = accuracy_score(label_test.cpu(), pred_test.cpu())
            f1w = f1_score(label_test.cpu(), pred_test.cpu(), average='weighted')
            f1m = f1_score(label_test.cpu(), pred_test.cpu(), average='macro')
            print(f"Test ACC: {acc:.5f}, F1 weighted : {f1w:.5f}, F1 macro: {f1m:.5f}")
            if acc > global_acc:
                global_acc = acc
                best_eval = {"acc": acc, 
                                "f1w":f1w, 
                                "f1m":f1m}

    return best_eval, exp_name



def model_test_3(model, data_test, graph):
    model.eval()
    with torch.no_grad():
        _, pred, _, _,_, _, _, _, _ , _= model(graph=graph, input_data=data_test)
    return pred
