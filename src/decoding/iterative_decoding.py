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

from src.data.vocabulary import PAD
from src.decoding.utils import tile_batch
from .beam_search import beam_search_return_BOS as beam_search
from .beam_search import nmt_lm_fusion_beam_search
from src.data.vocabulary import BOS, EOS, PAD, UNK
from src.models.base import NMTModel
from .utils import mask_scores, tensor_gather_helper


def rerank_beams_v2(beams, scores):
    """
        beams: [batch, beam, len]
        scores: [batch, beam], negative logprobs
    """
    _, reranked_ids = torch.sort(scores, dim=1, descending=False)  # [batch, beam]
    tiled_reranked_ids = reranked_ids.unsqueeze(-1).repeat(1, 1, beams.size(-1))
    reranked_beams = torch.gather(beams, dim=1,
                                  index=tiled_reranked_ids)
    return reranked_beams


def get_TM(src_lang, tgt_lang):
    return "{}2{}".format(src_lang, tgt_lang)


def initialize_translation(src, model, s2t_TM, src_lang, latent_size, beam_size, alpha):
    """ 
    Args:
        src (torch.LongTensor): [batch_size, src_len]
        s2t_TM: TM

    Returns:
        (torch.LongTensor): [batch_size, tgt_len]

    """
    latent_ = src.new_zeros([src.size(0), model.args.latent_size], dtype=torch.float32)
    model.set_latent(latent_)

    beam_y_ = beam_search(s2t_TM, beam_size, src.size(1) + 10, src, alpha)

    model.delete_latent()
    return beam_y_[:, :1, :].squeeze(1)


def mirror_iterative_decoding_v2(
    model,
    src, tgt=None, 
    src_lang="src", tgt_lang="tgt",
    reranking=False,
    iterations=3,
    beam_size=4, 
    alpha=0.6, beta=0.0, gamma=0.0):
    

    # prepare
    src_LM, tgt_LM = model.LMs[src_lang], model.LMs[tgt_lang]
    s2t_TM, t2s_TM = model.TMs[get_TM(src_lang, tgt_lang)], model.TMs[get_TM(tgt_lang, src_lang)]
    
    # 0. get initial translation
    tgt_ = initialize_translation(src, model, s2t_TM,
                                  src_lang=src_lang,
                                  latent_size=model.args.latent_size,
                                  beam_size=beam_size, alpha=alpha)
    src_beam = tile_batch(src, beam_size)

    for _ in range(iterations):
        # 1. infer latent
        latent_ = model.forward_inference_model(
            src, tgt_, is_sampling=False, src_lang=src_lang)["latent"]
        model.set_latent(latent_)

        # 2.1 decoding: get translation candidates
        # tgt_: [batch, beam, len_y]
        if beta <= 0:  # do not use tgt_LM for decoding
            tgt_ = beam_search(
                s2t_TM, 
                src_seqs=src,
                beam_size=beam_size, 
                max_steps=int(src.size(1)*1.2),
                alpha=alpha)
        else:
            tgt_ = nmt_lm_fusion_beam_search(
                nmt_model=s2t_TM,
                lm_model=tgt_LM,
                src_seqs=src,
                beam_size=beam_size,
                max_steps=int(src.size(1)*1.2),
                alpha=alpha,
                beta=beta,
                with_bos=True)

        # 2.2 reconstructive reranking 
        if reranking:
            tgt_beam = tgt_.reshape(-1, tgt_.size(-1)) # [bsz*beam, len_y]
            latent_beam = model.forward_inference_model(
                src_beam, tgt_beam, is_sampling=False, src_lang=src_lang)["latent"]
            
            rec_score_TM = model.score_translation_model(
                t2s_TM,
                src=tgt_beam, tgt=src_beam,
                latent=latent_beam,
                src_lang=tgt_lang, tgt_lang=src_lang)

            if gamma <= 0: # do not use src_LM for reconstructive reranking
                rec_score_LM = 0.
            else:
                rec_score_LM = model.score_language_model(
                    src_LM,
                    tgt=src_beam,
                    latent=latent_beam,
                    tgt_lang=src_lang)
            
            rec_score = (rec_score_TM + gamma*rec_score_LM).view(-1, beam_size) # [bsz, beam]
            reranked_tgt = rerank_beams_v2(tgt_, rec_score) # [bsz, beam, len_y]
            tgt_ = reranked_tgt
        
        # [batch, 1, len_y]. the best intermediate translation 
        tgt_ = tgt_[:, :1, :].squeeze(1)

    model.delete_latent()
    return tgt_[:, None, 1:].contiguous()
        
