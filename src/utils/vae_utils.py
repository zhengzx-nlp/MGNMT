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

import numpy as np

from src.data.vocabulary import BOS, PAD, EOS, UNK


def kl_anneal_function(anneal_function, step, k, x0, max_v=1.0):
    if anneal_function == "fixed":
        ret = 1.0
    elif anneal_function == 'logistic':
        ret = float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'sigmoid':
        ret = float(1 / (1 + np.exp(0.001 * (x0 - step))))
    elif anneal_function == 'negative-sigmoid':
        ret = float(1 / (1 + np.exp(-0.001 * (x0 - step))))
    elif anneal_function == 'linear':
        ret = min(1, step / x0)
    return min(max_v, ret)


def wd_anneal_function(unk_max, anneal_function, step, k, x0):
    return unk_max * kl_anneal_function(anneal_function, step, k, x0)


def unk_replace(input_sequence, dropout):
    if dropout > 0.:
        prob = torch.rand(input_sequence.size())
        if torch.cuda.is_available(): prob = prob.cuda()
        prob[(input_sequence.data - BOS) * (input_sequence.data - PAD) * (
                input_sequence.data - EOS) == 0] = 1
        decoder_input_sequence = input_sequence.clone()
        decoder_input_sequence[prob < dropout] = UNK
        return decoder_input_sequence
    return input_sequence
