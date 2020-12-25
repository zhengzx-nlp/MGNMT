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

from src.utils.logging import INFO


def tensor_dict_to_float_dict(**tensors):
    return {kk: vv.item() if isinstance(vv, torch.Tensor) else vv for kk, vv in tensors.items()}


def get_tensor_dict_str(**tensors):
    float_dict = tensor_dict_to_float_dict(**tensors)
    return ", ".join(["{}: {:.3f}".format(kk, vv)
                      for kk, vv in float_dict.items()])


def print_tensor_dict(**tensors):
    INFO("\n" + get_tensor_dict_str(**tensors))


def reduce_tensors_through_batch_dim(**tensors)-> {}:
    for k, t in tensors.items():
        if isinstance(t, torch.Tensor):
            tensors[k] = t.sum(0)
    return tensors
