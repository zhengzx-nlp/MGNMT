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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.vocabulary import PAD
from src.decoding.utils import tile_batch, tensor_gather_helper
from src.models.base import NMTModel
from src.modules.basic import BottleLinear as Linear
from src.modules.embeddings import Embeddings
from src.modules.sublayers import PositionwiseFeedForward, MultiHeadedAttention
from src.utils import nest


def get_attn_causal_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.

    :param seq: Input sequence.
        with shape [batch_size, time_steps, dim]
    '''
    assert seq.dim() == 3
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


class TransformerDecoderBlock(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1):
        super().__init__()

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)
        self.ctx_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)
        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def compute_cache(self, enc_output):
        return self.ctx_attn.compute_cache(enc_output, enc_output)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None,
                enc_attn_cache=None, self_attn_cache=None):
        # Args Checks
        input_batch, input_len, _ = dec_input.size()

        contxt_batch, contxt_len, _ = enc_output.size()

        input_norm = self.layer_norm_1(dec_input)
        all_input = input_norm

        query, _, self_attn_cache = self.slf_attn(all_input, all_input, input_norm,
                                                  mask=slf_attn_mask, self_attn_cache=self_attn_cache)

        query = self.dropout(query) + dec_input

        query_norm = self.layer_norm_2(query)
        mid, attn, enc_attn_cache = self.ctx_attn(enc_output, enc_output, query_norm,
                                                  mask=dec_enc_attn_mask, enc_attn_cache=enc_attn_cache)

        output = self.pos_ffn(self.dropout(mid) + query)

        return output, attn, self_attn_cache, enc_attn_cache


class TransformerDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_tgt_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dim_per_head=None, dropout=0.1):

        super().__init__()

        self.n_head = n_head
        self.num_layers = n_layers
        self.d_model = d_model

        self.embeddings = Embeddings(n_tgt_vocab, d_word_vec,
                                     dropout=dropout, add_position_embedding=True)

        self.block_stack = nn.ModuleList([
            TransformerDecoderBlock(d_model=d_model, d_inner_hid=d_inner_hid,
                                    n_head=n_head, dropout=dropout,
                                    dim_per_head=dim_per_head)
            for _ in range(n_layers)])

        self.out_layer_norm = nn.LayerNorm(d_model)

        self._dim_per_head = dim_per_head

    @property
    def dim_per_head(self):
        if self._dim_per_head is None:
            return self.d_model // self.n_head
        else:
            return self._dim_per_head

    def forward(self, tgt_seq, enc_output, enc_mask, enc_attn_caches=None, self_attn_caches=None):

        batch_size, tgt_len = tgt_seq.size()

        query_len = tgt_len
        key_len = tgt_len

        src_len = enc_output.size(1)

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt_seq)

        if self_attn_caches is not None:
            emb = emb[:, -1:].contiguous()
            query_len = 1

        # Decode mask
        dec_slf_attn_pad_mask = tgt_seq.detach().eq(PAD).unsqueeze(1).expand(batch_size, query_len, key_len)
        dec_slf_attn_sub_mask = get_attn_causal_mask(emb)

        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, query_len, src_len)

        output = emb
        new_self_attn_caches = []
        new_enc_attn_caches = []
        for i in range(self.num_layers):
            output, attn, self_attn_cache, enc_attn_cache = \
                self.block_stack[i](
                    output,
                    enc_output,
                    dec_slf_attn_mask,
                    dec_enc_attn_mask,
                    enc_attn_cache=enc_attn_caches[i]
                        if enc_attn_caches is not None else None,
                    self_attn_cache=self_attn_caches[i]
                        if self_attn_caches is not None else None)

            new_self_attn_caches += [self_attn_cache]
            new_enc_attn_caches += [enc_attn_cache]

        output = self.out_layer_norm(output)

        return output, new_self_attn_caches, new_enc_attn_caches


class VTransformerDecoder(TransformerDecoder):
    def __init__(self, n_tgt_vocab, n_layers=6, n_head=8,
                 d_word_vec=512, d_model=512, d_inner_hid=1024,
                 dropout=0.1, dim_per_head=None, latent_size=100):
        super().__init__(n_tgt_vocab, n_layers, n_head, d_word_vec, d_model,
                         d_inner_hid, dim_per_head, dropout)
        self.linear_inp = nn.Linear(d_word_vec + latent_size, d_model)

    def forward(self, tgt_seq, enc_output, enc_mask, latent, enc_attn_caches=None, self_attn_caches=None):

        batch_size, tgt_len = tgt_seq.size()

        query_len = tgt_len
        key_len = tgt_len

        src_len = enc_output.size(1)

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt_seq)

        if self_attn_caches is not None:
            emb = emb[:, -1:].contiguous()
            query_len = 1

        latent = latent.unsqueeze(1).repeat(1, emb.size(1), 1)
        inp = self.linear_inp(torch.cat([emb, latent], -1))

        # Decode mask
        dec_slf_attn_pad_mask = tgt_seq.detach().eq(PAD).\
            unsqueeze(1).expand(batch_size, query_len, key_len)
        dec_slf_attn_sub_mask = get_attn_causal_mask(emb)

        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, query_len, src_len)

        output = inp
        new_self_attn_caches = []
        new_enc_attn_caches = []
        for i in range(self.num_layers):
            output, attn, self_attn_cache, enc_attn_cache = \
                self.block_stack[i](
                    output,
                    enc_output,
                    dec_slf_attn_mask,
                    dec_enc_attn_mask,
                    enc_attn_cache=enc_attn_caches[i]
                        if enc_attn_caches is not None else None,
                    self_attn_cache=self_attn_caches[i]
                        if self_attn_caches is not None else None)

            new_self_attn_caches += [self_attn_cache]
            new_enc_attn_caches += [enc_attn_cache]

        output = self.out_layer_norm(output)

        return output, new_self_attn_caches, new_enc_attn_caches
