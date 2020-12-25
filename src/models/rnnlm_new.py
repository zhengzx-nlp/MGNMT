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

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.init as my_init
from src.data.vocabulary import PAD
from src.decoding.utils import tile_batch, tensor_gather_helper
from src.modules.cgru import CGRUCell
from src.modules.criterions import NMTCriterion
from src.modules.embeddings import Embeddings
from src.modules.generator import Generator
from src.modules.rnn import RNN
from src.utils.vae_utils import unk_replace


class RNNLM(nn.Module):
    def __init__(self, n_words, input_size, hidden_size, dropout, embed=None):
        super().__init__()
        self.hidden_size = hidden_size

        # Use PAD
        if embed is not None:
            self.embed = embed 
        else:
            self.embed = Embeddings(num_embeddings=n_words,
                                    embedding_dim=input_size,
                                    dropout=dropout,
                                    add_position_embedding=False)

        self.gru = RNN(type="gru", batch_first=True,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        bidirectional=False)

        self.linear_logit = nn.Linear(hidden_size+input_size, input_size)

        self.dropout = nn.Dropout(dropout)
        self.generator = Generator(
            n_words, input_size,
            padding_idx=PAD,
            shared_weight=self.embed.embeddings.weight)

    def forward(self, seq, emb=None, hidden=None):
        seq_mask = seq.detach().eq(PAD)
        if emb is None:
            emb = self.embed(seq)

        # if self.latent_type == "input":
        #     if emb.dim() > 2:
        #         z = z.clone().unsqueeze(1).repeat(1, emb.size(1), 1)
        #     inp = torch.cat([emb, z], -1)
        # elif self.latent_type == "init":
        #     inp = emb
        #     if hidden is None and emb.dim() > 2:  # training and eval
        #         hidden = self.linear_init(z)

        hidden, _ = self.gru(emb, seq_mask,
                             h_0=hidden.unsqueeze(0) if hidden is not None else None)

        logits = F.tanh(self.linear_logit(torch.cat([hidden, emb], -1)))
        logits = self.dropout(logits)
        logprobs = self.generator(logits, True)

        return {"logprob": logprobs, "hidden": hidden}

    # def get_loss(self, y, z, normalization=1., reduce=True):
    #     y_inp = y[:, :-1].contiguous()
    #     y_label = y[:, 1:].contiguous()

    #     if self.training and self.word_dropout:
    #         y_inp = unk_replace(y_inp, self.step_unk_rate)

    #     ret = self.forward(y_inp, z)

    #     return {
    #         "logprobs": ret["logprobs"],
    #         "nll_loss": self.nll_criterion(
    #             ret["logprobs"], y_label, reduce=reduce) / normalization
    #     }

    def init_decoder(self, batch_size, expand_size=1):
        dec_init = torch.zeros(batch_size, self.hidden_size)
        if torch.cuda.is_available():
            dec_init = dec_init.cuda()

        assert hasattr(self, "latent")
        latent = self.latent
        
        if expand_size > 1:
            dec_init = tile_batch(dec_init, expand_size)
            latent = tile_batch(latent, multiplier=expand_size)

        return {"dec_hiddens": dec_init, "latent": latent}

    def decode(self, seq, dec_states, emb=None, log_probs=True):
        dec_hiddens = dec_states['dec_hiddens']
        latent = dec_states["latent"]

        final_word_indices = seq[:, -1:].contiguous()
        if emb is None:
            emb = self.forward_embedding(final_word_indices, latent)

        new_dec_states = self.forward(final_word_indices,
                                      emb=emb,
                                      hidden=dec_hiddens)

        scores = new_dec_states["logprob"].squeeze(1)
        next_hiddens = new_dec_states["hidden"].squeeze(1)

        dec_states["dec_hiddens"] = next_hiddens

        return scores, dec_states

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):
        dec_hiddens = dec_states["dec_hiddens"]
        batch_size = dec_hiddens.size(0) // beam_size

        dec_hiddens = tensor_gather_helper(gather_indices=new_beam_indices,
                                           gather_from=dec_hiddens,
                                           batch_size=batch_size,
                                           beam_size=beam_size,
                                           gather_shape=[batch_size * beam_size, -1])

        dec_states['dec_hiddens'] = dec_hiddens

        return dec_states


    def forward_embedding(self, seq, latent, add_pos=True):
        emb = self.embed(seq)
        if add_pos:
            emb = self.pos_embed(emb) 

        # concat embeddings with latent and do a linear transformation.
        _latent = latent.unsqueeze(1).repeat(1, emb.size(1), 1)
        var_emb = self.var_inp_map(torch.cat([emb, _latent], -1)) 
        return var_emb