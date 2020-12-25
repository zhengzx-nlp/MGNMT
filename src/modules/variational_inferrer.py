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
from torch import nn

from src.encoder.rnn_encoder import RNNEncoder
from src.modules.attention import ScaledDotProductAttention
from src.modules.embeddings import Embeddings
from src.modules.sublayers import MultiHeadedAttention


class _VariationalInferrer(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, d_word_vec, d_model,
                 d_latent, src_embed=None, tgt_embed=None):
        super().__init__()

        self.n_src_vocab = n_src_vocab
        self.n_tgt_vocab = n_tgt_vocab
        self.d_word_vec = d_word_vec
        self.d_model = d_model
        self.d_latent = d_latent

        self.src_embed = Embeddings(num_embeddings=n_src_vocab,
                                    embedding_dim=d_word_vec)
        self.tgt_embed = Embeddings(num_embeddings=n_tgt_vocab,
                                    embedding_dim=d_word_vec)
        if src_embed is not None:
            self.src_embed.embeddings.weight = \
                src_embed.embeddings.weight
        if tgt_embed is not None:
            self.tgt_embed.embeddings.weight = \
                tgt_embed.embeddings.weight

        self.infer_latent2mean = nn.Linear(d_model * 4, d_latent)
        self.infer_latent2logv = nn.Linear(d_model * 4, d_latent)

        self.should_swap = False

    def forward(self, x, y, is_sampling=True, stop_grad_input=True):
        batch_size = x.size(0)

        if self.should_swap:
            x, y = y, x

        bilingual_inf = self.encode(x, y, stop_grad_input)

        mean = self.infer_latent2mean(bilingual_inf)
        logv = self.infer_latent2logv(bilingual_inf)

        if is_sampling:
            std = torch.exp(0.5 * logv)
            z = mean.new_tensor(torch.randn([batch_size, self.d_latent]))
            z = z * std + mean
        else:
            z = mean

        return {
            "mean": mean,
            "logv": logv,
            "latent": z
        }

    def encode(self, x, y, stop_grad_emb=True):
        raise NotImplementedError

    def share_parameters(self, reverse_inferrer, swap=True):
        raise NotImplementedError


class RNNInferrer(_VariationalInferrer):
    def __init__(self, n_src_vocab, n_tgt_vocab, d_word_vec, d_model, d_latent,
                 src_embed=None, tgt_embed=None):
        super().__init__(n_src_vocab, n_tgt_vocab, d_word_vec, d_model, d_latent,
                         src_embed, tgt_embed)

        self.infer_enc_x = RNNEncoder(
            n_src_vocab, d_word_vec, d_model, embeddings=self.src_embed)
        self.infer_enc_y = RNNEncoder(
            n_tgt_vocab, d_word_vec, d_model, embeddings=self.tgt_embed)
        # self.infer_enc_x.embeddings = self.src_embed
        # self.infer_enc_y.embeddings = self.tgt_embed

    @staticmethod
    def _pool(_h, _m):
        _no_pad_mask = 1.0 - _m.float()
        _ctx_mean = (_h * _no_pad_mask.unsqueeze(2)).sum(1) / \
                    _no_pad_mask.unsqueeze(2).sum(1)
        return _ctx_mean

    def encode(self, x, y, stop_grad_input=True):
        x_emb, y_emb = self.src_embed(x), self.tgt_embed(y)
        if stop_grad_input:
            x_emb, y_emb = x_emb.detach(), y_emb.detach()

        enc_x, x_mask = self.infer_enc_x(x, x_emb)
        enc_y, y_mask = self.infer_enc_y(y, y_emb)

        bilingual_inf = torch.cat([self._pool(enc_x, x_mask),
                                   self._pool(enc_y, y_mask)], -1)
        return bilingual_inf

    def share_parameters(self, reverse_inferrer: _VariationalInferrer, swap=True):
        self.should_swap = swap
        self.src_embed, self.tgt_embed = \
            reverse_inferrer.src_embed, reverse_inferrer.tgt_embed
        self.infer_enc_x, self.infer_enc_y = \
            reverse_inferrer.infer_enc_x, reverse_inferrer.infer_enc_y
        self.infer_latent2mean, self.infer_latent2logv = \
            reverse_inferrer.infer_latent2mean, reverse_inferrer.infer_latent2logv


class InteractiveRNNInferrer(RNNInferrer):
    def __init__(self, n_src_vocab, n_tgt_vocab,
                 d_word_vec, d_model, d_latent,
                 src_embed=None, tgt_embed=None):
        super().__init__(n_src_vocab, n_tgt_vocab, d_word_vec,
                         d_model, d_latent,
                         src_embed, tgt_embed)
        # self.attn1 = MultiHeadedAttention(
        #     d_model*2, 1, dropout=0.)
        # self.attn2 = MultiHeadedAttention(
        #     d_model*2, 1, dropout=0.)
        self.attn1 = ScaledDotProductAttention(d_model*2, 0.)
        self.attn2 = ScaledDotProductAttention(d_model*2, 0.)

        self.infer_latent2mean = nn.Linear(d_model * 8, d_latent)
        self.infer_latent2logv = nn.Linear(d_model * 8, d_latent)

    def encode(self, x, y, stop_grad_input=True):
        x_emb, y_emb = self.src_embed(x), self.tgt_embed(y)
        if stop_grad_input:
            x_emb, y_emb = x_emb.detach(), y_emb.detach()

        enc_x, x_mask = self.infer_enc_x(x, x_emb)
        enc_y, y_mask = self.infer_enc_y(y, y_emb)

        # attn_x2y, _, _ = self.attn1(
        #     enc_y, enc_y, enc_x,
        #     mask=y_mask[:, None, :].repeat(1, enc_x.size(1), 1))
        # attn_y2x, _, _ = self.attn2(
        #     enc_x, enc_x, enc_y,
        #     mask=x_mask[:, None, :].repeat(1, enc_y.size(1), 1))
        attn_x2y, _ = self.attn1(
            enc_x, enc_y, enc_y,
            attn_mask=y_mask[:, None, :].repeat(1, enc_x.size(1), 1))
        attn_y2x, _ = self.attn2(
            enc_y, enc_x, enc_x,
            attn_mask=x_mask[:, None, :].repeat(1, enc_y.size(1), 1))

        bilingual_inf = torch.cat(
            [self._pool(enc_x, x_mask),
             self._pool(enc_y, y_mask),
             self._pool(attn_x2y, x_mask),
             self._pool(attn_y2x, y_mask)],
            -1)

        return bilingual_inf
