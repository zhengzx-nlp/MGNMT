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
    def __init__(self, n_words, d_word_vec, d_model, dropout, **kwargs):
        super().__init__()

        self.d_model = d_model
        # Use PAD
        self.embeddings = Embeddings(num_embeddings=n_words,
                                     embedding_dim=d_word_vec,
                                     dropout=0.0,
                                     add_position_embedding=False)

        self.gru = RNN(type="gru", batch_first=True, input_size=d_word_vec,
                       hidden_size=d_model,
                       bidirectional=False)

        self.linear_logit = nn.Linear(d_model + d_word_vec, d_word_vec)

        self.dropout = nn.Dropout(dropout)
        self.generator = Generator(
            n_words, d_word_vec,
            padding_idx=PAD,
            shared_weight=self.embeddings.embeddings.weight)

    def forward(self, x, hidden=None):
        x_mask = x.detach().eq(PAD)
        emb = self.embeddings(x)

        # ctx, _ = self.gru(emb, x_mask)

        ctx, _ = self.gru(emb, x_mask,
                          h_0=hidden.unsqueeze(0)
                            if hidden is not None else None)

        logits = F.tanh(self.linear_logit(torch.cat([ctx, emb], -1)))
        logits = self.dropout(logits)
        logprobs = self.generator(logits, True)

        return {"logprobs": logprobs, "hidden": hidden}

    def init_decoder(self, batch_size, expand_size=1):
        dec_init = torch.zeros(batch_size, self.d_model)
        if torch.cuda.is_available():
            dec_init = dec_init.cuda()

        if expand_size > 1:
            dec_init = tile_batch(dec_init, expand_size)
        return {"dec_hiddens": dec_init}

    def decode(self, tgt_seq, dec_states, log_probs=True):
        dec_hiddens = dec_states['dec_hiddens']

        final_word_indices = tgt_seq[:, -1:].contiguous()

        new_dec_states = self.forward(final_word_indices,
                                      hidden=dec_hiddens)

        scores = new_dec_states["logprobs"].squeeze(1)
        next_hiddens = new_dec_states["hidden"].squeeze(1)

        dec_states = {"dec_hiddens": next_hiddens}

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
