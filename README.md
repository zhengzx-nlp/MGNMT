# MGNMT

This repo is the implementation of [Mirror-Generative Neural Machine Translation](https://openreview.net/forum?id=HkxQRTNYPH).

## 0. Preparation
MGNMT is implemented based on NJUNMT. 
As for data preprocessing and others please refer to [README-NJUNMT](README-NJUNMT.md)

This repo has been tested with `python 3.7` and `pytorch 1.2.0` on 2080ti and V100.

We should prepare following datasets
1. parallel bilingual training dataset with bpe codes, vocabs
2. dev and test datasets
3. (optional) non-paralell bilingual training dataset, i.e., monolingual dataset for source and target language

Please see configurations in `configs/ed/` for example.


## 1. Training
### Parallel Data Only
```bash
sh train_mgnmt_iwslt_de2en_ed.sh
```
This script uses configuration in `configs/ed/iwslt_de2en_mgnmt.yaml`

### Using Non-Parallel Data
```bash
sh train_mgnmt_np_iwslt_de2en_ed.sh
```
This script uses configuration in `configs/ed/iwslt_de2en_mgnmt_np.yaml`

- for domain adaptation, using `NP_train_mode` as `"finetune"` may be better
- for other scenarios, using `"hybrid"` to enable hybrid training of both P and NP data
- it would be better to start using NP data after the model has been well trained on parallel data, which should probably take a bit larger `NP_warmup_step`.


### Multi-GPU Training
Specify the environment variable `CUDA_VISIBLE_DEVICES`. For example:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
```
Currently this repo only support multi gpus on a single node. 

## 2. Generation
```bash
sh translate_mgnmt.sh
```
Note that we have both src2tgt and tgt2src translation direction, so we should specify `--src_lang` to either `"src"` or `"tgt"` to specify the expected translation direction.

- Use `beta` to control the weight of decoding target LM fusion. 
- Use `--reranking` to enable reconstructive reranking. 
- Use `gamma` to control the weight of source LM in reconstructive reranking. 

- `--batch_size` determines batch size based on the number of tokens.

## 3. Others
### Important files
`src/tasks/mgnmt.py`

`src/models/mgnmt.py`

`src/decoding/iterative_decoding.py`

`src/modules/variational_inferrer.py`

and configs in `./configs/`

### Tips
1. How well the inference model learns affects the performance of MGNMT. So having a proper KL loss is important. However, KL annealing is VERY tricky. Try to adjust `x0` (KL annealing step) to find an optimal KL loss that is neither large nor small too much. A KL loss of ~3 to 6 works well in practice. 

2. For the best training efficiency, it is highly recommended that do not use reranking and LM fusion in back-translation or bleu_validation. Plus you may not know the optimal weights to interpolate LMs. You can find the best weights in inference after you get the training done.

### TODO
- continue refactoring codebase
- pretrained models
- hyperparameters and settings for other experiments in the paper

## Citation
```
@inproceedings{zheng2020mirror,
    title={Mirror-Generative Neural Machine Translation},
    author={Zaixiang Zheng and Hao Zhou and Shujian Huang and Lei Li and Xin-Yu Dai and Jiajun Chen},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=HkxQRTNYPH}
}

```