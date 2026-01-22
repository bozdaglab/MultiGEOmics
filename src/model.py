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
from model_config import ADNI_ORDER


class FeatureAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2), nn.ReLU(), nn.Linear(in_dim * 2, in_dim)
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
            self.feature_attn_modules = nn.ModuleDict(
                {rel: FeatureAttention(in_dim) for rel in self.rel_names}
            )
        else:
            self.norm = {
                key: nn.LayerNorm(shape[1]).to(device)
                for key, shape in key_shape.items()
            }
            self.feature_attn_modules = nn.ModuleDict(
                {key: FeatureAttention(shape[1]) for key, shape in key_shape.items()}
            )

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
        return output, attn_scores.sum(dim=1)

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

        keys = list(h.keys())
        idx = 1
        while idx < len(updated_attention_embeddings):
            encoder_inp = updated_attention_embeddings[keys[idx - 1]]
            decoder_inp = updated_attention_embeddings[keys[idx]]
            Q = self.split_heads(decoder_inp.unsqueeze(-1))
            K = self.split_heads(encoder_inp.unsqueeze(-1))
            V = self.split_heads(encoder_inp.unsqueeze(-1))
            attn_output, attn_scores = self.scaled_dot_product_attention(
                Q, K, V, mask=None
            )
            output = self.combine_heads(attn_output)
            try:
                final_output = self.norm[keys[idx]](self.dropout(output.squeeze(-1)) + decoder_inp)
            except TypeError:
                final_output = self.norm(self.dropout(output.squeeze(-1)) + decoder_inp)
            updated_attention_embeddings[keys[idx]] = final_output
            if h[keys[idx]].shape[1] == attn_scores.shape[2]:
                pass
            else:
                attn_scores = attn_scores.permute(0, 2, 1)
            feature_attention[f"{keys[idx - 1]}_{keys[idx]}"] = attn_scores
            idx += 1
        return (
            updated_attention_embeddings,
            feature_attention,
        )



class VAE(nn.Module):
    def __init__(self, input_size, hid_size, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
        )
        
        self.mu = nn.Linear(hid_size, latent_size)
        self.var = nn.Linear(hid_size, latent_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, input_size)
        )
        
    def reparameterization(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.rand_like(std)
        return mu + eps * std
    
    def forward(self, input_data):
        encoder_embeddings = self.encoder(input_data)
        mu = self.mu(encoder_embeddings)
        var = self.var(encoder_embeddings)
        z = self.reparameterization(mu, var)
        return self.decoder(z), mu, var, z
        
class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        return mu.squeeze(2), logvar.squeeze(2)

class GCNLatent(nn.Module):
    def __init__(self, x_dim, z_dim, nonLinear):
        super(GCNLatent, self).__init__()
        
        self.latentnet = torch.nn.ModuleList([
            nn.Linear(x_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            Gaussian(z_dim, 1)
        ])

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)#和vae好像还是不一样的？还是技巧，哪个好
        z = mu + noise * std#为啥不是exp(std) 答：forward那块加了

        return z
    
    def latent(self, x):

        for layer in self.latentnet:
            x = layer(x)
        return x

    def forward(self, x, ):
       
        mu, logvar = self.latent(x.float())
        var = torch.exp(logvar)
        z = self.reparameterize(mu, var)
        output = {'lmean'  : mu, 'lvar': var, 'lvalue': z,}
        return output
    

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
        learn_param: bool = True
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
        self.learn_param = learn_param
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
        
        # self.conv1 = nn.ModuleDict(
        # {
        #     rel: dglnn.GraphConv(shape[1], shape[1])
        #     for rel, shape in self.omics_shapes.items()
        # }
        # )
        # self.conv2 = nn.ModuleDict(
        #     {
        #         rel: dglnn.GraphConv(shape[1], shape[1])
        #         for rel, shape in self.omics_shapes.items()
        #     }
        # )
       
        if self.dataset in [
            DataEnum.BRCA_M.name,
            DataEnum.KIPAN.name,
            DataEnum.ROSMAP_M.name,
            DataEnum.LGG.name,
        ]:
            self.encoder_decoder = nn.ModuleDict(
                {
                    rel: VAE(input_size=shape[1], hid_size=256, latent_size=64)
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

        first_hop_embeddings = {
            key: self.conv1[key](graph[key], value) for key, value in input_data.items()
        }
        first_hop_embeddings = self.correct_shape(first_hop_embeddings)
        (
            first_hop_att_embeddings,
            first_feature_attention,
        ) = self.attentionencoder(first_hop_embeddings)
        if self.args.dataset != DataEnum.ADNI.name:
            first_hop_rev_embeddings = sort_data_order(
                dataset=self.args.dataset, train_data=first_hop_embeddings, forwards=False
                )
            first_hop_rev_embeddings = self.correct_shape(first_hop_rev_embeddings)
            (first_hop_att_embeddings_rev, 
            first_omics_attention_rev, 
            ) = self.attentionencoder(first_hop_rev_embeddings)
            first_con = {k: first_hop_embeddings[k] * first_hop_att_embeddings[k] 
                        for k in first_hop_embeddings.keys()}
            second_con = {k: first_hop_embeddings[k] * first_hop_att_embeddings_rev[k] 
                        for k in first_hop_embeddings.keys()}
            first_hop_att_embeddings = {k: first_con[k] + second_con[k] 
                                    for k in first_con.keys()}

        second_hop_embeddings = {
            etyoe: self.conv2[etyoe](graph[etyoe], first_hop_att_embeddings[etyoe])
            for etyoe in input_data.keys()
        }
        if self.two_level_attention:
            second_hop_embeddings_correct_shape = self.correct_shape(
                second_hop_embeddings
            )
            (second_hop_att_embeddings, 
            #  second_omics_attention, 
             second_feature_attention) = self.attentionencoder(
                second_hop_embeddings_correct_shape
            )

            second_hop_rev_embeddings = sort_data_order(dataset=self.args.dataset, train_data=second_hop_embeddings, forwards=False)
            second_hop_rev_embeddings = self.correct_shape(second_hop_rev_embeddings)
            (second_hop_att_embeddings_rev, 
            #  second_omics_attention_rev, 
             _) = self.attentionencoder(second_hop_rev_embeddings)
            

            first_con = {k: second_hop_embeddings[k] * second_hop_att_embeddings[k] for k in second_hop_embeddings.keys()}
            second_con = {k: second_hop_embeddings[k] * second_hop_att_embeddings_rev[k] for k in second_hop_embeddings.keys()}
            fin_embds = {k: first_con[k] + second_con[k] for k in first_con.keys()}

            return (fin_embds, 
                    # first_omics_attention, 
                    first_feature_attention,
                    # first_omics_attention_rev, 
                    # first_feature_attention_rev,
                    # second_omics_attention, 
                    second_feature_attention,
                    # second_omics_attention_rev, 
                    # second_feature_attention_rev
            )
        try:
            return (second_hop_embeddings,
                    # first_omics_attention, 
                    first_feature_attention,
                    first_omics_attention_rev, 
                    # first_feature_attention_rev,
                        # "second_omics_attention", 
                        # "second_feature_attention",
                        # "second_omics_attention_rev", 
                        # "second_feature_attention_rev"
                # second_hop_embeddings,
                # # self.correct_shape(second_hop_embeddings),
                # omics_attention,
                # feature_attention,
                # # omics_attention_rev, 
                # # feature_attention_rev
            )
        except UnboundLocalError:
            return (second_hop_embeddings,
                    # first_omics_attention, 
                    first_feature_attention,
                    "first_omics_attention_rev", 
                    # first_feature_attention_rev,
                        # "second_omics_attention", 
                        # "second_feature_attention",
                        # "second_omics_attention_rev", 
                        # "second_feature_attention_rev"
                # second_hop_embeddings,
                # # self.correct_shape(second_hop_embeddings),
                # omics_attention,
                # feature_attention,
                # # omics_attention_rev, 
                # # feature_attention_rev
            )

    def forward(
        self, graph: DGLHeteroGraph, input_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        (
            second_hop_embeddings,
            # first_omics_attention, 
            first_feature_attention,
            # first_omics_attention_rev, 
            first_feature_attention_rev,
            # second_omics_attention, 
            # second_feature_attention,
            # second_omics_attention_rev, 
            # second_feature_attention_rev
        ) = self.message_passings_embeddings(graph, input_data)
        if self.reverse_attention:
            reverse_input_data = sort_data_order(
                dataset=self.args.dataset, train_data=input_data, forwards=False
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
        if self.dataset in [
            DataEnum.BRCA_M.name,
            DataEnum.KIPAN.name,
            DataEnum.ROSMAP_M.name,
            DataEnum.LGG.name,
        ]:
            recons = defaultdict()
            mus = defaultdict()
            logvars = defaultdict()
            embed = defaultdict()
            for key, value in second_hop_embeddings.items():
                recons[key], mus[key], logvars[key], embed[key] = self.encoder_decoder[key](value)
            if isinstance(recons, dict):
                second_hop_embeddings = list(recons.values())
            elif isinstance(recons, Tensor):
                second_hop_embeddings = list(recons)
            out_embeddings = self.lin_transpose(torch.concat(second_hop_embeddings, dim=-1))
            return (
                out_embeddings,
                self.label_classifier(out_embeddings),
                recons,
                # first_omics_attention, 
                first_feature_attention,
                # first_omics_attention_rev, 
                first_feature_attention_rev,
                # second_omics_attention, 
                "second_feature_attention",
                # second_omics_attention_rev, 
                "second_feature_attention_rev",
                mus, logvars, recons
            )
        else:
            if isinstance(second_hop_embeddings, dict):
                second_hop_embeddings = list(second_hop_embeddings.values())
            elif isinstance(second_hop_embeddings, Tensor):
                second_hop_embeddings = list(second_hop_embeddings)
            out_embeddings = self.lin_transpose(torch.concat(second_hop_embeddings, dim=-1))
            return (
                out_embeddings,
                self.label_classifier(out_embeddings),
                # first_omics_attention, 
                first_feature_attention,
                # first_omics_attention_rev, 
                first_feature_attention_rev,
                # second_omics_attention, 
                # second_feature_attention,
                # second_omics_attention_rev, 
                # "second_feature_attention_rev"
            )
