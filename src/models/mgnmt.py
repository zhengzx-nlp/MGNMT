# -*- coding: UTF-8 -*- 
# 
# MIT License
#
# Copyright (c) 2018 the xnmt authors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#       author: Zaixiang Zheng
#       contact: zhengzx@nlp.nju.edu.cn 
#           or zhengzx.142857@gmail.com
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F

from src.utils.configs import args_to_dict, dict_to_args
from src.utils.tensor_ops import reduce_tensors_through_batch_dim
from src.utils.vae_utils import wd_anneal_function, kl_anneal_function, unk_replace
from src.models.transformer_new import Transformer
from src.models.rnnlm_new import RNNLM
from src.modules.variational_inferrer import RNNInferrer, InteractiveRNNInferrer
from src.modules.criterions import NMTCriterion
from src.modules.embeddings import Embeddings
from src.modules.position_embedding import PositionalEmbedding


def change(m):
    return {"src": "tgt", "tgt": "src"}

class MirrorGNMT(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.kl_weight = 1.

        self.build_model(args)
    
    def build_model(self, args):
        self._build_embeddings(args)
        self._build_language_models(args)
        self._build_translation_models(args)
        self._build_inference_model(args)
        self._build_criterion(args)
        
    
    def _build_embeddings(self, args):
       src_embed = Embeddings(num_embeddings=args.n_src_vocab,
                              embedding_dim=args.d_word_vec,
                              dropout=args.dropout,
                              add_position_embedding=True)
       tgt_embed = Embeddings(num_embeddings=args.n_tgt_vocab,
                              embedding_dim=args.d_word_vec,
                              dropout=args.dropout,
                              add_position_embedding=True)
       src_embed.add_position_embedding = False
       tgt_embed.add_position_embedding = False
       self.embeds = nn.ModuleDict({"src": src_embed, "tgt": tgt_embed})
       self.pos_embed = PositionalEmbedding(demb=args.d_word_vec, dropout=args.dropout)

    def _build_language_models(self, args):
        lm = {}
        lm["src"] = RNNLM(args.n_src_vocab,
                          input_size=args.d_word_vec,
                          hidden_size=args.d_model,
                          dropout=args.dropout,
                          embed=self.embeds["src"])
        lm["tgt"] = RNNLM(args.n_tgt_vocab,
                          input_size=args.d_word_vec,
                          hidden_size=args.d_model,
                          dropout=args.dropout,
                          embed=self.embeds["tgt"])
        self.LMs = nn.ModuleDict(lm)

    def _build_translation_models(self, args):
        tm = {}
        tm["src2tgt"] = Transformer(
            args.n_src_vocab, args.n_tgt_vocab,
            src_embed=self.embeds["src"], tgt_embed=self.embeds["tgt"],
            n_layers=args.n_layers, n_head=args.n_head, 
            d_word_vec=args.d_word_vec, d_model=args.d_model, d_inner_hid=args.d_inner_hid,
            dropout=args.dropout,
            tie_source_target_embedding=args.tie_source_target_embedding,
            tie_input_output_embedding=args.tie_input_output_embedding)
       
        tm["tgt2src"] = Transformer(
            args.n_tgt_vocab, args.n_src_vocab,
            src_embed=self.embeds["tgt"], tgt_embed=self.embeds["src"],
            n_layers=args.n_layers, n_head=args.n_head, 
            d_word_vec=args.d_word_vec, d_model=args.d_model, d_inner_hid=args.d_inner_hid,
            dropout=args.dropout,
            tie_source_target_embedding=args.tie_source_target_embedding,
            tie_input_output_embedding=args.tie_input_output_embedding)
        self.TMs = nn.ModuleDict(tm)

        self.TMs["src2tgt"].pos_embed = self.pos_embed
        self.TMs["tgt2src"].pos_embed = self.pos_embed
        self.LMs["src"].pos_embed = self.pos_embed
        self.LMs["tgt"].pos_embed = self.pos_embed

    def _build_inference_model(self, args):
        # TODO: fix embedding 
        self.inferrer = RNNInferrer(
            args.n_src_vocab, args.n_tgt_vocab,
            args.d_word_vec, args.d_model, args.latent_size,
            src_embed=self.embeds["src"],
            tgt_embed=self.embeds["tgt"])

        self.var_inp_maps = nn.ModuleDict({
            "src": nn.Linear(args.d_word_vec+args.latent_size, args.d_word_vec),
            "tgt": nn.Linear(args.d_word_vec+args.latent_size, args.d_word_vec)
        })

        self.TMs["src2tgt"].var_inp_maps = {
            "src": self.var_inp_maps["src"],
            "tgt": self.var_inp_maps["tgt"]
        }
        self.TMs["tgt2src"].var_inp_maps = {
            "src": self.var_inp_maps["tgt"],
            "tgt": self.var_inp_maps["src"]
        }
        self.LMs["src"].var_inp_map = self.var_inp_maps["src"]
        self.LMs["tgt"].var_inp_map = self.var_inp_maps["tgt"]

    def _build_criterion(self, args):
        self.nll_criterion = NMTCriterion(label_smoothing=self.args.label_smoothing)
        
    def forward_inference_model(self, src, tgt, is_sampling=True, src_lang="src"):
        # src_emb, tgt_emb = self.embeds["src"](src), self.embeds["tgt"](tgt)
        if src_lang == "tgt":
            src, tgt = tgt, src

        inferred = self.inferrer(
            src, tgt,
            is_sampling=is_sampling,
            stop_grad_input=self.args.inferrer_stop_grad_input)

        # concat embeddings with latent and do a linear transformation.
        # latent = inferred["latent"].unsqueeze(1)
        # var_src_emb = self.var_inp_maps["src"](torch.cat([src_emb, latent], -1))
        # var_tgt_emb = self.var_inp_maps["tgt"](torch.cat([tgt_emb, latent], -1))

        return {
            "mean": inferred["mean"],
            "logv": inferred["logv"],
            "latent": inferred["latent"],
            # "var_src_emb": var_src_emb,
            # "var_tgt_emb": var_tgt_emb
        }

    def forward_embedding(self, seq, latent, lang="src", add_pos=True):
        emb = self.embeds[lang](seq)
        if add_pos:
            emb = self.pos_embed(emb)
        
        # concat embeddings with latent and do a linear transformation.
        _latent = latent.unsqueeze(1).repeat(1, emb.size(1), 1)
        var_emb = self.var_inp_maps[lang](
            torch.cat([emb, _latent], -1)) 
        return var_emb
        
    def forward_translation_model(self, TM, src, tgt, src_emb, tgt_emb):
        # tgt_emb_inp = tgt_emb[:, :-1, :].contiguous()
        logprob = TM.forward(src, tgt, src_emb, tgt_emb)
        return {"logprob": logprob}
    
    def score_translation_model(self, TM, src, tgt, latent, src_lang="src", tgt_lang="tgt"):
        tgt_inp = tgt[:, :-1].contiguous()
        tgt_label = tgt[:, 1:].contiguous()

        src_emb = self.forward_embedding(src, latent, lang=src_lang, add_pos=True)
        tgt_emb = self.forward_embedding(tgt_inp, latent, lang=tgt_lang, add_pos=True)
        
        logprob = self.forward_translation_model(TM, src, tgt_inp, src_emb, tgt_emb)["logprob"]
        nll_score = self.nll_criterion(logprob, tgt_label, reduce=False) # [bsz,]
        return nll_score
    
    def forward_language_model(self, LM, tgt, tgt_emb):
        # tgt_emb_inp = tgt_emb[:, :-1, :].contiguous() 
        LM_ret = LM.forward(tgt, tgt_emb)
        return {"logprob": LM_ret["logprob"]}
    
    def score_language_model(self, LM, tgt, latent, tgt_lang="tgt"):
        tgt_inp = tgt[:, :-1].contiguous()
        tgt_label = tgt[:, 1:].contiguous()

        tgt_emb = self.forward_embedding(tgt_inp, latent, lang=tgt_lang, add_pos=True)
        
        logprob = self.forward_language_model(LM, tgt_inp, tgt_emb)["logprob"]
        nll_score = self.nll_criterion(logprob, tgt_label, reduce=False) # [bsz,]
        return nll_score
    
    def prepare_word_dropout(self, inp):
        if self.training:
            return unk_replace(inp, self.args.unk_rate)
        else:
            return inp

    def forward(self, mode="all", *inputs, **kwargs):
        if mode == "all":
            return self.compute_loss(*inputs, **kwargs)
        elif mode == "partial":
            return self.compute_loss_partial(*inputs, **kwargs)
    
    def forward_all(self, src, tgt):
        # 1. inference model
        inferred = self.forward_inference_model(
            src, tgt, self.training)
        latent = inferred.pop("latent")
        
        # 2. embedding
        # UNK dropout
        src_dropped = self.prepare_word_dropout(src)
        tgt_dropped = self.prepare_word_dropout(tgt)

        src_emb = self.forward_embedding(src_dropped, latent, lang="src", add_pos=True)
        tgt_emb = self.forward_embedding(tgt_dropped, latent, lang="tgt", add_pos=True)
        
        src_inp, tgt_inp = src_dropped[:, :-1].contiguous(), tgt_dropped[:, :-1].contiguous()
        src_inp_emb, tgt_inp_emb = src_emb[:, :-1].contiguous(), tgt_emb[:, :-1].contiguous()

        # 3. logprobs of LMs and TMs
        logprobs = {}

        logprobs["src"] = self.forward_language_model(
            self.LMs["src"], src_inp, src_inp_emb)["logprob"]
        logprobs["tgt"] = self.forward_language_model(
            self.LMs["tgt"], tgt_inp, tgt_inp_emb)["logprob"]

        logprobs["src2tgt"] = self.forward_translation_model(
            self.TMs["src2tgt"], 
            src, tgt_inp, src_emb, tgt_inp_emb)["logprob"]
        logprobs["tgt2src"] = self.forward_translation_model(
            self.TMs["tgt2src"], 
            tgt, src_inp, tgt_emb, src_inp_emb)["logprob"]
        
        return {"logprobs": logprobs, "inferred": inferred}

        
    def forward_partial(self, src_, tgt, src_lang="src", tgt_lang="tgt"):
        LM_name, TM_name = tgt_lang, "{}2{}".format(src_lang, tgt_lang)
        tgt_LM = self.LMs[tgt_lang]
        s2t_TM = self.TMs[TM_name]
        
        # 1. inference model
        inferred = self.forward_inference_model(
            src_, tgt, self.training, src_lang=src_lang)
        latent = inferred.pop("latent")
        
        # 2. embedding
        # UNK dropout
        src_dropped = self.prepare_word_dropout(src_)
        src_emb = self.forward_embedding(src_dropped, latent, lang=src_lang, add_pos=True)

        tgt_dropped = self.prepare_word_dropout(tgt)
        tgt_inp = tgt_dropped[:, :-1].contiguous()
        tgt_inp_emb = self.forward_embedding(tgt_inp, latent, lang=tgt_lang, add_pos=True)
        # 3. logprobs of LMs and TMs
        logprobs = {}

        logprobs[LM_name] = self.forward_language_model(
            tgt_LM, tgt_inp, tgt_inp_emb)["logprob"]

        logprobs[TM_name] = self.forward_translation_model(
            s2t_TM, 
            src_, tgt_inp, src_emb, tgt_inp_emb)["logprob"]
        
        return {"logprobs": logprobs, "inferred": inferred}

    def compute_loss(self, src, tgt, step):
        model_ret = self.forward_all(src, tgt)
        logprobs = model_ret["logprobs"]
        inferred = model_ret["inferred"]

        # 1. nll loss 
        nll_losses = {}

        src_label = src[:, 1:].contiguous()
        tgt_label = tgt[:, 1:].contiguous()

        nll_losses["LM_src"] = self.nll_criterion(logprobs["src"], src_label)
        nll_losses["LM_tgt"] = self.nll_criterion(logprobs["tgt"], tgt_label)
        nll_losses["TM_src2tgt"] = self.nll_criterion(logprobs["src2tgt"], tgt_label)
        nll_losses["TM_tgt2src"] = self.nll_criterion(logprobs["tgt2src"], src_label) 
        nll_losses["total_nll"] = 0.5 * sum(nll_losses.values())

        # 2. KL loss
        kl_losses = self.compute_kl_loss(
            inferred['mean'],
            inferred['logv'],
            step=step)

        ELBO = kl_losses["kl_loss"] + nll_losses["total_nll"]
        final_loss = kl_losses["kl_term"] + nll_losses["total_nll"]
        
        return {"Loss": final_loss, "ELBO": ELBO, **nll_losses, **kl_losses} 

    def compute_loss_partial(self, src_, tgt, step, src_lang="src", tgt_lang="tgt"):
        LM_name, TM_name = tgt_lang, "{}2{}".format(src_lang, tgt_lang)

        model_ret = self.forward_partial(src_, tgt, src_lang, tgt_lang)
        logprobs = model_ret["logprobs"]
        inferred = model_ret["inferred"]

        # 1. nll loss 
        nll_losses = {}

        tgt_label = tgt[:, 1:].contiguous()

        nll_losses["LM_%s" % LM_name] = self.nll_criterion(logprobs[LM_name], tgt_label)
        nll_losses["TM_%s" % TM_name] = self.nll_criterion(logprobs[TM_name], tgt_label)
        # nll_losses["total_nll"] = 0.5 * sum(nll_losses.values())
        nll_losses["total_nll"] = sum(nll_losses.values())

        # 2. KL loss
        kl_losses = self.compute_kl_loss(
            inferred['mean'],
            inferred['logv'],
            step=step)

        ELBO = kl_losses["kl_loss"] + nll_losses["total_nll"]
        final_loss = kl_losses["kl_term"] + nll_losses["total_nll"]
        
        return {"Loss": final_loss, "ELBO": ELBO, **nll_losses, **kl_losses} 

    def get_kl_weight(self, step):
        if self.args.step_kl_weight is None:
            return kl_anneal_function(
                self.args.anneal_function, step,
                self.args.k, self.args.x0, self.args.max_kl_weight)
        else:
            return self.args.step_kl_weight

    def compute_kl_loss(self, mean, logv, step, normalization=1):
        kl_weight = self.get_kl_weight(step)
        self.kl_weight = kl_weight * self.args.kl_factor

        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        kl_loss = kl_loss / normalization

        kl_item = kl_loss * self.kl_weight
        return {"kl_loss": kl_loss, "kl_term": kl_item}

    # for easily manipulate decoding
    def set_latent(self, latent):
        """
            latent: [bsz, d_model]
        """
        self.TMs["src2tgt"].latent = latent
        self.TMs["tgt2src"].latent = latent
        self.LMs["src"].latent = latent
        self.LMs["tgt"].latent = latent
    
    def delete_latent(self):
        delattr(self.TMs["src2tgt"], "latent")
        delattr(self.TMs["tgt2src"], "latent")
        delattr(self.LMs["src"], "latent")
        delattr(self.LMs["tgt"], "latent")