import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.heterograph import DGLHeteroGraph
from torch import Tensor

from enum_holder import DataEnum
from helper import sort_data_order


class FeatureAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        weights = torch.sigmoid(self.attn(x))
        x_attended = weights * x
        return x_attended, weights


class LabelClassifier(nn.Module):
    def __init__(self, inp_dim, out_dim, in_feats_double):
        super().__init__()
        self.in_feats_double = in_feats_double
        if self.in_feats_double:
            inp_dim = inp_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_features=inp_dim, out_features=int(inp_dim / 2)),
            nn.ELU(),
            nn.Linear(in_features=int(inp_dim / 2), out_features=int(inp_dim / 4)),
            nn.ELU(),
            nn.Linear(in_features=int(inp_dim / 4), out_features=out_dim),
        )

    def forward(self, out_embeddings):
        return self.mlp(out_embeddings)


class SemanticAttention(nn.Module):
    def __init__(
        self,
        num_relations,
        in_dim,
        dim_a,
        rel_names,
        key_shape,
        dataset,
        device,
        dropout=0.0,
    ):
        super(SemanticAttention, self).__init__()
        self.num_relations = num_relations
        self.device = device
        self.dataset = dataset
        self.in_dim = in_dim
        self.dim_a = dim_a
        self.rel_names = rel_names
        self.num_heads = self.embed_dim = 1
        self.d_k = 1
        self.dropout = nn.Dropout(dropout)

        if isinstance(self.in_dim, int):
            self.norm = nn.LayerNorm(self.in_dim)
            self.weights_s1 = nn.Parameter(
                torch.FloatTensor(self.num_relations, self.in_dim, self.dim_a)
            )
            self.feature_attn_modules = nn.ModuleDict(
                {rel: FeatureAttention(in_dim) for rel in self.rel_names}
            )
            self.weights_s2 = nn.Parameter(
                torch.FloatTensor(self.num_relations, self.dim_a, self.num_relations)
            )
            self.reset_parameters()
        else:
            self.norms = {
                key: nn.LayerNorm(shape[1]).to(device)
                for key, shape in key_shape.items()
            }
            self.weights_s1 = {
                key: nn.Parameter(torch.FloatTensor(1, shape[1], self.dim_a).to(device))
                for key, shape in key_shape.items()
            }

            self.feature_attn_modules = nn.ModuleDict(
                {key: FeatureAttention(shape[1]) for key, shape in key_shape.items()}
            )
            self.weights_s2 = nn.Parameter(torch.FloatTensor(1, self.dim_a, 1))
            self.reset_parameters_mult()

    def reset_parameters_mult(self):
        gain = nn.init.calculate_gain("tanh")
        for param in self.weights_s1.values():
            nn.init.xavier_uniform_(param.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def reset_parameters(self):
        gain = nn.init.calculate_gain("tanh")
        nn.init.xavier_uniform_(self.weights_s1.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def split_heads(self, x):
        try:
            batch_size, seq_length, _ = x.size()
        except ValueError:
            batch_size, seq_length, _ = x.unsqueeze(0).size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores.masked_fill_(mask == 0, float("-inf"))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output, attn_scores

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return (
            x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        )

    def forward(self, h, return_attn=True):
        updated_attention_embeddings = defaultdict()
        feature_attention = defaultdict()
        for omic in h.keys():
            (
                updated_attention_embeddings[omic],
                feature_attention[omic],
            ) = self.feature_attn_modules[omic](h[omic])

        if self.dataset in [
            DataEnum.ADNI.name,
            DataEnum.TCGA_GBM.name,
            DataEnum.TCGA_BRCA.name,
            DataEnum.AML.name,
            DataEnum.BLCA.name,
            DataEnum.BRCA.name,
            DataEnum.LIHC.name,
            DataEnum.PRAD.name,
            DataEnum.WT.name,
        ]:
            omics_att = torch.stack(
                list(
                    {
                        key: torch.sigmoid(
                            torch.matmul(value, self.weights_s1[key].to(self.device))
                        )
                        for key, value in updated_attention_embeddings.items()
                    }.values()
                )
            )
            omics_attention = F.softmax(
                torch.matmul(omics_att, self.weights_s2), dim=1
            ).permute(0, 2, 1, 3)
            omics_attention = self.dropout(omics_attention)
            updated_embed = {
                key: value.unsqueeze(1) * omics_attention[idx]
                for idx, (key, value) in enumerate(updated_attention_embeddings.items())
            }
            keys = list(h.keys())
            idx = 1
            if self.dataset in DataEnum.ADNI.name:
                encoder_inp = updated_embed["snps"].permute(0, 2, 1)
                decoder_inp = updated_embed["bile"].permute(0, 2, 1)
                Q = self.split_heads(decoder_inp)
                K = self.split_heads(encoder_inp)
                V = self.split_heads(encoder_inp)
                attn_output, attn_scores = self.scaled_dot_product_attention(
                    Q, K, V, mask=None
                )
                output = self.combine_heads(attn_output)
                final_output = self.norms["bile"](
                    self.dropout(output.squeeze(-1)).to(self.device)
                    + decoder_inp.squeeze(-1).to(self.device)
                )
                updated_attention_embeddings["bile"] = final_output
                feature_attention[f"snps_bile"] = attn_scores

                encoder_inp = updated_embed["snps"].permute(0, 2, 1)
                decoder_inp = updated_embed["lipids"].permute(0, 2, 1)
                Q = self.split_heads(decoder_inp)
                K = self.split_heads(encoder_inp)
                V = self.split_heads(encoder_inp)
                attn_output, attn_scores = self.scaled_dot_product_attention(
                    Q, K, V, mask=None
                )
                output = self.combine_heads(attn_output)
                final_output = self.norms["lipids"](
                    self.dropout(output.squeeze(-1)).to(self.device)
                    + decoder_inp.squeeze(-1).to(self.device)
                )
                updated_attention_embeddings["lipids"] = final_output
                feature_attention[f"snps_lipids"] = attn_scores
            else:
                while idx < len(updated_embed):
                    encoder_inp = updated_embed[keys[idx - 1]].permute(0, 2, 1)
                    decoder_inp = updated_embed[keys[idx]].permute(0, 2, 1)
                    Q = self.split_heads(decoder_inp)
                    K = self.split_heads(encoder_inp)
                    V = self.split_heads(encoder_inp)
                    attn_output, attn_scores = self.scaled_dot_product_attention(
                        Q, K, V, mask=None
                    )
                    output = self.combine_heads(attn_output)
                    final_output = self.norms[keys[idx]](
                        self.dropout(output.squeeze(-1)).to(self.device)
                        + decoder_inp.squeeze(-1).to(self.device)
                    )
                    updated_attention_embeddings[keys[idx]] = final_output
                    feature_attention[f"{keys[idx - 1]}_{keys[idx]}"] = attn_scores
                    idx += 1
        else:
            stacked_embeddings = torch.stack(
                list(updated_attention_embeddings.values())
            )
            omics_attention = F.softmax(
                torch.matmul(
                    torch.sigmoid(torch.matmul(stacked_embeddings, self.weights_s1)),
                    self.weights_s2,
                ),
                dim=0,
            ).permute(1, 0, 2)
            omics_attention = self.dropout(omics_attention)
            keys = list(h.keys())
            stacked_embeddings = torch.matmul(
                omics_attention, stacked_embeddings.permute(1, 0, 2)
            )
            idx = 1
            while idx < stacked_embeddings.shape[1]:
                encoder_inp = stacked_embeddings[:, idx - 1, :]
                decoder_inp = stacked_embeddings[:, idx, :]
                Q = self.split_heads(decoder_inp.unsqueeze(-1))
                K = self.split_heads(encoder_inp.unsqueeze(-1))
                V = self.split_heads(encoder_inp.unsqueeze(-1))
                attn_output, attn_scores = self.scaled_dot_product_attention(
                    Q, K, V, mask=None
                )
                output = self.combine_heads(attn_output)
                final_output = self.norm(self.dropout(output.squeeze(-1)) + decoder_inp)
                updated_attention_embeddings[keys[idx]] = final_output
                feature_attention[f"{keys[idx - 1]}_{keys[idx]}"] = attn_scores
                idx += 1
        return (
            updated_attention_embeddings,
            omics_attention if return_attn else None,
            feature_attention,
        )


class MultiGraphGCN(nn.Module):
    def __init__(
        self,
        hidden_feats: List[int],
        rel_names: List[str],
        num_patients: int,
        num_class: int,
        stack_types: str,
        hid_emb: int,
        args: Any,
        combination: Dict[str, Union[str, float, int, bool]],
        reverse_attention: bool,
        two_level_attention: bool,
        omics_shapes: Dict[str, Tuple[int]],
        device: torch.device,
    ):

        super().__init__()
        self.dataset = args.dataset
        self.two_level_attention = two_level_attention
        self.args = args
        self.device = device
        self.omics_shapes = {
            key: val for key, val in omics_shapes.items() if len(val) > 1
        }
        self.hidden_feats = hidden_feats
        self.num_omics = len(rel_names)
        self.num_patients = num_patients
        self.reverse_attention = reverse_attention
        self.num_class = num_class
        self.stack_types = stack_types

        self.label_classifier = LabelClassifier(
            inp_dim=hid_emb, out_dim=num_class, in_feats_double=False
        )

        self.conv1 = nn.ModuleDict(
            {
                rel: dglnn.SAGEConv(shape[1], shape[1], combination["aggregator_type"])
                for rel, shape in self.omics_shapes.items()
            }
        )
        self.conv2 = nn.ModuleDict(
            {
                rel: dglnn.SAGEConv(shape[1], shape[1], combination["aggregator_type"])
                for rel, shape in self.omics_shapes.items()
            }
        )

        self.attentionencoder = SemanticAttention(
            num_relations=self.num_omics,
            in_dim=hidden_feats,
            dropout=combination["dropout"],
            dim_a=20,
            rel_names=rel_names,
            key_shape=self.omics_shapes,
            dataset=self.dataset,
            device=device,
        )

        self.lin_transpose = nn.Linear(
            sum([val[1] for val in self.omics_shapes.values()]), hid_emb
        )

    def correct_shape(
        self, embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if len(embeddings[list(embeddings.keys())[0]].shape) > 2:
            return {key: value.mean(dim=1) for key, value in embeddings.items()}
        else:
            return embeddings

    def message_passings_embeddings(
        self, graph: DGLHeteroGraph, input_data: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:

        embeddings = {
            key: self.conv1[key](graph[key], value) for key, value in input_data.items()
        }
        embeddings = self.correct_shape(embeddings)
        (
            first_hop_embeddings,
            omics_attention,
            feature_attention,
        ) = self.attentionencoder(embeddings)

        second_hop_embeddings = {
            etyoe: self.conv2[etyoe](graph[etyoe], first_hop_embeddings[etyoe])
            for etyoe in input_data.keys()
        }
        if self.two_level_attention:
            second_hop_embeddings_correct_shape = self.correct_shape(
                second_hop_embeddings
            )
            fin_embeddings, omics_attention, feature_attention = self.attentionencoder(
                second_hop_embeddings_correct_shape
            )

            fin_embds = {etyoe: fin_embeddings[etyoe] for etyoe in input_data.keys()}
            return fin_embds, omics_attention, feature_attention

        return (
            self.correct_shape(second_hop_embeddings),
            omics_attention,
            feature_attention,
        )

    def forward(
        self, graph: DGLHeteroGraph, input_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        (
            second_hop_embeddings,
            omics_attention_forward,
            feature_attention_forward,
        ) = self.message_passings_embeddings(graph, input_data)
        if self.reverse_attention:
            reverse_input_data = sort_data_order(
                args=self.args, train_data=input_data, forwards=False
            )
            (
                second_hop_embeddings_reverse,
                omics_attention_reverse,
                feature_attention_reverse,
            ) = self.message_passings_embeddings(graph, reverse_input_data)
            second_hop_embeddings = {
                key: torch.sum(
                    torch.stack(
                        [second_hop_embeddings[key], second_hop_embeddings_reverse[key]]
                    ),
                    dim=0,
                )
                for key in second_hop_embeddings.keys()
            }

        if isinstance(second_hop_embeddings, dict):
            second_hop_embeddings = list(second_hop_embeddings.values())
        elif isinstance(second_hop_embeddings, Tensor):
            second_hop_embeddings = list(second_hop_embeddings)
        out_embeddings = self.lin_transpose(
            torch.concat(second_hop_embeddings, dim=-1)
        )
        return (
            out_embeddings,
            self.label_classifier(out_embeddings),
            omics_attention_forward,
            feature_attention_forward,
        )
