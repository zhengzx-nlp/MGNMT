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

from src.data.vocabulary import PAD
from src.modules.embeddings import Embeddings
from src.modules.rnn import RNN


class RNNEncoder(nn.Module):
    def __init__(self,
                 n_words,
                 input_size,
                 hidden_size,
                 embeddings=None):
        super(RNNEncoder, self).__init__()

        # Use PAD
        self.embeddings = Embeddings(num_embeddings=n_words,
                                     embedding_dim=input_size,
                                     dropout=0.0,
                                     add_position_embedding=False)
        if embeddings is not None:
            self.embeddings.embeddings.weight = \
                embeddings.embeddings.weight
        self.gru = RNN(type="gru", batch_first=True,
                       input_size=input_size, hidden_size=hidden_size,
                       bidirectional=True)

    def forward(self, x, emb=None):
        """
        :param x: Input sequence.
            with shape [batch_size, seq_len, input_size]
        """
        x_mask = x.detach().eq(PAD)

        if emb is None:
            emb = self.embeddings(x)

        ctx, _ = self.gru(emb, x_mask)

        return ctx, x_mask


class VRNNEncoder(RNNEncoder):
    def __init__(self, n_words, input_size, hidden_size, latent_size,
                 latent_type="input"):
        super().__init__(n_words, input_size, hidden_size)
        self.latent_type = latent_type
        if latent_type == "input":
            self.gru = RNN(type="gru", batch_first=True,
                           input_size=input_size+latent_size,
                           hidden_size=hidden_size,
                           bidirectional=True)
        elif latent_type == "init":
            self.gru = RNN(type="gru", batch_first=True,
                           input_size=input_size,
                           hidden_size=hidden_size,
                           bidirectional=True)
            self.hidden_size = hidden_size
            self.linear_init = nn.Linear(latent_size, hidden_size * 2)

    def forward(self, x, z):
        batch_size, x_len = x.size()
        x_mask = x.detach().eq(PAD)
        emb = self.embeddings(x)

        if self.latent_type == "input":
            z = z.clone().unsqueeze(1).repeat(1, x_len, 1)
            inp = torch.cat([emb, z], -1)
            ctx, _ = self.gru(inp, x_mask)
        elif self.latent_type == "init":
            inp = emb
            h_0_z = self.linear_init(z)\
                .view(batch_size, 2, self.hidden_size)\
                .transpose(0, 1)
            ctx, _ = self.gru(inp, x_mask, h_0=h_0_z)

        return ctx, x_mask
