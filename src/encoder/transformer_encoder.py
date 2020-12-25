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

import torch.nn as nn
import torch

from src.data.vocabulary import PAD
from src.modules.embeddings import Embeddings
from src.modules.sublayers import PositionwiseFeedForward, MultiHeadedAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)

        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        input_norm = self.layer_norm(enc_input)
        context, _, _ = self.slf_attn(input_norm, input_norm, input_norm, slf_attn_mask)
        out = self.dropout(context) + enc_input

        return self.pos_ffn(out)


class TransformerEncoder(nn.Module):

    def __init__(
            self, n_src_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1, dim_per_head=None):
        super().__init__()

        self.num_layers = n_layers
        self.embeddings = Embeddings(num_embeddings=n_src_vocab,
                                     embedding_dim=d_word_vec,
                                     dropout=dropout,
                                     add_position_embedding=True)
        self.block_stack = nn.ModuleList(
            [TransformerEncoderBlock(
                d_model=d_model, d_inner_hid=d_inner_hid,
                n_head=n_head, dropout=dropout,
                dim_per_head=dim_per_head)
             for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src_seq):
        # Word embedding look up
        batch_size, src_len = src_seq.size()

        emb = self.embeddings(src_seq)

        enc_mask = src_seq.detach().eq(PAD)
        enc_slf_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, src_len, src_len)

        out = emb

        for i in range(self.num_layers):
            out = self.block_stack[i](out, enc_slf_attn_mask)

        out = self.layer_norm(out)

        return out, enc_mask


class VTransformerEncoder(TransformerEncoder):
    def __init__(self, n_src_vocab, n_layers=6, n_head=8,
                 d_word_vec=512, d_model=512, d_inner_hid=1024,
                 dropout=0.1, dim_per_head=None, latent_size=100):
        super().__init__(n_src_vocab, n_layers, n_head, d_word_vec, d_model,
                         d_inner_hid, dropout, dim_per_head)
        self.linear_inp = nn.Linear(d_word_vec + latent_size, d_model)

    def forward(self, src_seq, latent):
        # Word embedding look up
        batch_size, src_len = src_seq.size()

        emb = self.embeddings(src_seq)

        # concat with latent and do a linear transformation.
        latent = latent.unsqueeze(1).repeat(1, src_len, 1)
        inp = self.linear_inp(torch.cat([emb, latent], -1))

        enc_mask = src_seq.detach().eq(PAD)
        enc_slf_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, src_len, src_len)

        out = inp

        for i in range(self.num_layers):
            out = self.block_stack[i](out, enc_slf_attn_mask)

        out = self.layer_norm(out)

        return out, enc_mask
