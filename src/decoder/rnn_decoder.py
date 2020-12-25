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
from src.decoding.utils import tile_batch
from src.modules.cgru import CGRUCell
from src.modules.criterions import NMTCriterion
from src.modules.embeddings import Embeddings
from src.modules.generator import Generator
from src.modules.rnn import RNN
from src.utils.vae_utils import unk_replace


class RNNDecoder(nn.Module):
    def __init__(self, n_words, input_size, hidden_size):
        super().__init__()

        # Use PAD
        self.embeddings = Embeddings(num_embeddings=n_words,
                                     embedding_dim=input_size,
                                     dropout=0.0,
                                     add_position_embedding=False)

        self.gru = RNN(type="gru", batch_first=True, input_size=input_size,
                       hidden_size=hidden_size,
                       bidirectional=False)

        self.linear_logit = nn.Linear(hidden_size + input_size, input_size)

        self.generator = Generator(
            n_words, input_size,
            padding_idx=PAD,
            shared_weight=self.embeddings.embeddings.weight)

    def forward(self, x):
        """
        :param x: Input sequence.
            with shape [batch_size, seq_len, input_size]
        """
        x_mask = x.detach().eq(PAD)
        emb = self.embeddings(x)

        ctx, _ = self.gru(emb, x_mask)

        logits = F.tanh(self.linear_logit(torch.cat([ctx, emb], -1)))
        logprobs = self.generator(logits, True)

        return logprobs


class VRNNDecoder(nn.Module):
    def __init__(self, n_words, input_size, hidden_size, latent_size,
                 latent_type="input", word_dropout=False):
        super().__init__()
        self.latent_type = latent_type
        # Use PAD
        self.embeddings = Embeddings(num_embeddings=n_words,
                                     embedding_dim=input_size,
                                     dropout=0.0,
                                     add_position_embedding=False)

        if self.latent_type == "input":
            self.gru = RNN(type="gru", batch_first=True,
                           input_size=input_size + latent_size,
                           hidden_size=hidden_size,
                           bidirectional=False)
        elif self.latent_type == "init":
            self.gru = RNN(type="gru", batch_first=True,
                           input_size=input_size,
                           hidden_size=hidden_size,
                           bidirectional=False)
            self.linear_init = nn.Linear(latent_size, hidden_size)

        self.linear_logit = nn.Linear(hidden_size + input_size, input_size)

    def init_decoder(self, latent, expand_size=1):
        if self.latent_type == "init":
            dec_init = self.linear_init(latent)
        elif self.latent_type == "input":
            dec_init = latent.new_zeros(latent.size(0), self.hidden_size)

        if expand_size > 1:
            dec_init = tile_batch(dec_init, expand_size)
            latent = tile_batch(latent, expand_size)

        return {"dec_hiddens": dec_init, "latent": latent}

    def forward(self, x, z, hidden=None):
        x_mask = x.detach().eq(PAD)
        emb = self.embeddings(x)

        if self.latent_type == "input":
            if emb.dim() > 2:
                z = z.clone().unsqueeze(1).repeat(1, emb.size(1), 1)
            inp = torch.cat([emb, z], -1)
        elif self.latent_type == "init":
            inp = emb
            assert hidden is not None

        hidden, _ = self.gru(inp, x_mask,
                             h_0=hidden.unsqueeze(0) if hidden is not None else None)

        logits = F.tanh(self.linear_logit(torch.cat([hidden, emb], -1)))

        return logits


class AttnDecoder(nn.Module):
    def __init__(self,
                 n_words,
                 input_size,
                 hidden_size,
                 context_size,
                 bridge_type="mlp",
                 dropout_rate=0.0):
        super(AttnDecoder, self).__init__()

        self.bridge_type = bridge_type
        self.hidden_size = hidden_size
        self.context_size = context_size
        # self.context_size = hidden_size * 2

        self.embeddings = Embeddings(num_embeddings=n_words,
                                     embedding_dim=input_size,
                                     dropout=0.0,
                                     add_position_embedding=False)

        self.cgru_cell = CGRUCell(input_size=input_size,
                                  hidden_size=hidden_size,
                                  context_size=context_size)

        self.linear_input = nn.Linear(in_features=input_size,
                                      out_features=input_size)
        self.linear_hidden = nn.Linear(in_features=hidden_size,
                                       out_features=input_size)
        self.linear_ctx = nn.Linear(in_features=context_size,
                                    out_features=input_size)

        self.dropout = nn.Dropout(dropout_rate)

        self._reset_parameters()

        self._build_bridge()

    def _reset_parameters(self):
        my_init.default_init(self.linear_input.weight)
        my_init.default_init(self.linear_hidden.weight)
        my_init.default_init(self.linear_ctx.weight)

    def _build_bridge(self):
        if self.bridge_type == "mlp":
            self.linear_bridge = nn.Linear(in_features=self.context_size,
                                           out_features=self.hidden_size)
            my_init.default_init(self.linear_bridge.weight)
        elif self.bridge_type == "zero":
            pass
        else:
            raise ValueError("Unknown bridge type {0}".format(self.bridge_type))

    def init_decoder(self, context, mask, **kwargs):
        # Generate init hidden
        if self.bridge_type == "mlp":

            no_pad_mask = 1.0 - mask.float()
            ctx_mean = (context * no_pad_mask.unsqueeze(2)).sum(
                1) / no_pad_mask.unsqueeze(2).sum(1)
            dec_init = F.tanh(self.linear_bridge(ctx_mean))

        elif self.bridge_type == "zero":
            batch_size = context.size(0)
            dec_init = context.new(batch_size, self.hidden_size).zero_()
        else:
            raise ValueError("Unknown bridge type {0}".format(self.bridge_type))

        dec_cache = self.cgru_cell.compute_cache(context)

        return dec_init, dec_cache

    def forward(self, y, context, context_mask, hidden, one_step=False,
                cache=None, **kwargs):
        emb = self.embeddings(y)  # [batch_size, seq_len, dim]

        if one_step:
            (out, attn), hidden = self.cgru_cell(emb, hidden, context,
                                                 context_mask, cache)
        else:
            # emb: [batch_size, seq_len, dim]
            out = []
            attn = []

            for emb_t in torch.split(emb, split_size_or_sections=1, dim=1):
                (out_t, attn_t), hidden = self.cgru_cell(emb_t.squeeze(1),
                                                         hidden, context,
                                                         context_mask, cache)
                out += [out_t]
                attn += [attn_t]

            out = torch.stack(out).transpose(1, 0).contiguous()
            attn = torch.stack(attn).transpose(1, 0).contiguous()

        logits = self.linear_input(emb) + self.linear_hidden(
            out) + self.linear_ctx(attn)

        logits = F.tanh(logits)

        logits = self.dropout(logits)  # [batch_size, seq_len, dim]

        return logits, hidden


class VAttnDecoder(nn.Module):
    def __init__(self,
                 n_words,
                 input_size,
                 hidden_size,
                 context_size,
                 latent_size,
                 latent_type="input",
                 bridge_type="zero",
                 dropout_rate=0.0):
        super().__init__()

        self.bridge_type = bridge_type
        self.hidden_size = hidden_size
        self.context_size = context_size

        self.latent_size = latent_size
        self.latent_type = latent_type

        self.embeddings = Embeddings(num_embeddings=n_words,
                                     embedding_dim=input_size,
                                     dropout=0.0,
                                     add_position_embedding=False)

        if latent_type == "input":
            self.cgru_cell = CGRUCell(input_size=input_size + latent_size,
                                      hidden_size=hidden_size,
                                      context_size=context_size)
        elif latent_type == "init":
            self.cgru_cell = CGRUCell(input_size=input_size,
                                      hidden_size=hidden_size,
                                      context_size=context_size)

        self.linear_input = nn.Linear(in_features=input_size,
                                      out_features=input_size)
        self.linear_hidden = nn.Linear(in_features=hidden_size,
                                       out_features=input_size)
        self.linear_ctx = nn.Linear(in_features=context_size,
                                    out_features=input_size)

        self.dropout = nn.Dropout(dropout_rate)

        self._reset_parameters()

        self._build_bridge()

    def _reset_parameters(self):
        my_init.default_init(self.linear_input.weight)
        my_init.default_init(self.linear_hidden.weight)
        my_init.default_init(self.linear_ctx.weight)

    def _build_bridge(self):
        if self.latent_type == "init":
            self.linear_bridge = nn.Linear(in_features=self.latent_size,
                                           out_features=self.hidden_size)
            my_init.default_init(self.linear_bridge.weight)
        else:
            pass

    def init_decoder(self, context, mask, latent):
        # Generate init hidden
        if self.latent_type == "init":
            dec_init = F.tanh(self.linear_bridge(latent))
        else:
            batch_size = context.size(0)
            dec_init = context.new(batch_size, self.hidden_size).zero_()

        dec_cache = self.cgru_cell.compute_cache(context)

        return dec_init, dec_cache

    def forward(self, y, z, context, context_mask, hidden,
                one_step=False, cache=None):
        emb = self.embeddings(y)  # [batch_size, seq_len, dim]

        if self.latent_type == "input":
            if emb.dim() > 2:
                z = z.clone().unsqueeze(1).repeat(1, emb.size(1), 1)
            inp = torch.cat([emb, z], -1)
        elif self.latent_type == "init":
            inp = emb

        if one_step:
            (out, attn), hidden = self.cgru_cell(inp, hidden, context,
                                                 context_mask, cache)
        else:
            # emb: [batch_size, seq_len, dim]
            out = []
            attn = []

            for emb_t in torch.split(inp, split_size_or_sections=1, dim=1):
                (out_t, attn_t), hidden = self.cgru_cell(emb_t.squeeze(1),
                                                         hidden, context,
                                                         context_mask, cache)
                out += [out_t]
                attn += [attn_t]

            out = torch.stack(out).transpose(1, 0).contiguous()
            attn = torch.stack(attn).transpose(1, 0).contiguous()

        logits = self.linear_input(emb) + self.linear_hidden(
            out) + self.linear_ctx(attn)

        logits = F.tanh(logits)

        logits = self.dropout(logits)  # [batch_size, seq_len, dim]

        return logits, hidden
