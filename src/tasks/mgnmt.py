# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import random
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, ZipDataset
from src.data.vocabulary import Vocabulary
from src.decoding.iterative_decoding import mirror_iterative_decoding_v2
from src.metric.bleu_scorer import SacreBLEUScorer
from src.models import build_model
from src.optim import Optimizer
from src.optim.lr_scheduler import (
    ReduceOnPlateauScheduler,
    NoamScheduler,
    RsqrtScheduler,
)
from src.utils.common_utils import *
from src.utils.common_utils import calculate_parameter_stats
from src.utils.configs import default_configs, pretty_configs, dict_to_args
from src.utils.logging import *
from src.utils.moving_average import MovingAverage
from src.utils.tensor_ops import (
    tensor_dict_to_float_dict,
    print_tensor_dict,
    get_tensor_dict_str,
)
from src.utils import distributed_utils

BOS = Vocabulary.BOS
EOS = Vocabulary.EOS
PAD = Vocabulary.PAD


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


def load_model_parameters(path, map_location="cpu"):
    state_dict = torch.load(path, map_location=map_location)

    if "model" in state_dict:
        return state_dict["model"]
    return state_dict


def split_shard(*inputs, split_size=1):
    if split_size <= 1:
        yield inputs
    else:

        lengths = [len(s) for s in inputs[-1]]  #
        sorted_indices = np.argsort(lengths)

        # sorting inputs

        inputs = [[inp[ii] for ii in sorted_indices] for inp in inputs]

        # split shards
        total_batch = sorted_indices.shape[0]  # total number of batches

        if split_size >= total_batch:
            yield inputs
        else:
            shard_size = total_batch // split_size

            _indices = list(range(total_batch))[::shard_size] + [total_batch]

            for beg, end in zip(_indices[:-1], _indices[1:]):
                yield (inp[beg:end] for inp in inputs)


def prepare_data(seqs_x, seqs_y=None, cuda=False, batch_first=True):
    """
    Args:
        eval ('bool'): indicator for eval/infer.

    Returns:

    """

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):
        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype="int64")

        for ii in range(batch_size):
            x_np[ii, : sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [BOS] + s + [EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=PAD, cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x

    seqs_y = list(map(lambda s: [BOS] + s + [EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=PAD, cuda=cuda, batch_first=batch_first)

    return x, y


def compute_forward(
    model, seqs_x, seqs_y, eval=False, normalization=1.0, norm_by_words=False, step=0
):
    # y_inp = seqs_y[:, :-1].contiguous()
    # y_label = seqs_y[:, 1:].contiguous()
    #
    # words_norm = y_label.ne(PAD).float().sum(1)

    if not eval:
        model.train()
        # For training
        with torch.enable_grad():
            # losses = model.compute_loss(src=seqs_x, tgt=seqs_y, step=step)
            losses = model(src=seqs_x, tgt=seqs_y, step=step, mode="all")
            for kk in losses:
                losses[kk] = losses[kk] / normalization
        loss = losses["Loss"]
        torch.autograd.backward(loss)
        return losses
    else:
        model.eval()
        # For compute loss
        with torch.no_grad():
            losses = model(src=seqs_x, tgt=seqs_y, step=step, mode="all")
        losses = tensor_dict_to_float_dict(**losses)
        return losses


def train_step(model, optim, batch, training_configs, uidx):
    seqs_x, seqs_y = batch

    n_samples_t = len(seqs_x)
    n_words_t = sum(len(s) for s in seqs_y)

    is_oom = False
    losses = defaultdict(float)
    train_loss = 0.0
    optim.zero_grad()
    try:
        # gradient accumulation to simulate multi-gpu
        for seqs_x_t, seqs_y_t in split_shard(
            seqs_x, seqs_y, split_size=training_configs["update_cycle"]
        ):
            # Prepare data
            x, y = prepare_data(seqs_x_t, seqs_y_t, cuda=GlobalNames.USE_GPU)
            losses = compute_forward(
                model=model,
                seqs_x=x,
                seqs_y=y,
                eval=False,
                normalization=n_samples_t,
                norm_by_words=training_configs["norm_by_words"],
                step=uidx,
            )
            train_loss += losses["Loss"] / y.size(1)
        optim.step()
    except RuntimeError as e:
        if "out of memory" in str(e):
            WARN("| WARNING: ran out of memory, skipping batch")
            is_oom = True
            optim.zero_grad()
        else:
            raise

    losses["train_loss"] = train_loss
    return {
        "train_losses": losses,
        "n_samples_t": n_samples_t,
        "n_words_t": n_words_t,
        "is_oom": is_oom,
    }


def compute_forward_partial(
    model,
    src_,
    tgt,
    src_lang="src",
    tgt_lang="tgt",
    normalization=1.0,
    norm_by_words=False,
    step=0,
):
    model.train()
    # For training
    with torch.enable_grad():
        # losses = model.compute_loss_partial(
        losses = model(
            src_=src_,
            tgt=tgt,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            step=step,
            mode="partial",
        )
        for kk in losses:
            losses[kk] = losses[kk] / normalization
    loss = losses["Loss"]
    torch.autograd.backward(loss)
    return losses


def train_step_mono(
    model,
    optim,
    tgt_batch,
    training_configs,
    uidx,
    src_vocab,
    src_lang,
    tgt_lang,
    best_model=None,
):
    model_ = best_model if best_model is not None else model

    seqs_y = tgt_batch

    # best_model_ = best_model.backward_model if training_type == "forward" \
    #     else best_model.forward_model

    n_samples_t = len(seqs_y)
    n_words_t = sum(len(s) for s in seqs_y)

    is_oom = False
    losses = defaultdict(float)
    train_loss = 0.0

    try:
        # Prepare data
        for seqs_y_t in split_shard(
            seqs_y, split_size=training_configs["update_cycle"]
        ):
            y = prepare_data(list(seqs_y_t)[0], cuda=GlobalNames.USE_GPU)
            # back-translate pseudo translation
            with torch.no_grad():
                # [bsz, 1, len_y]
                pseudo_x = mirror_iterative_decoding_v2(
                    model_,
                    src=y,
                    src_lang=tgt_lang,
                    tgt_lang=src_lang,
                    reranking=False,
                    iterations=1,
                    beam_size=1,
                    alpha=1.0,
                )

                pseudo_x = pseudo_x.squeeze(1)
            if training_configs["train_nonparallel_options"]["NP_update_mode"] == "all":
                x = pseudo_x
                if src_lang != "src":
                    x, y = y, x
                # input ordering should always be "src", "tgt"
                losses = compute_forward(
                    model=model,
                    seqs_x=x,
                    seqs_y=y,
                    eval=False,
                    normalization=n_samples_t,
                    norm_by_words=training_configs["norm_by_words"],
                    step=uidx,
                )

            elif (
                training_configs["train_nonparallel_options"]["NP_update_mode"]
                == "partial"
            ):
                losses = compute_forward_partial(
                    model=model,
                    src_=pseudo_x,  # source is the synthetic BT data
                    tgt=y,  # target is the ground-truth monolingual data
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    normalization=n_samples_t,
                    norm_by_words=training_configs["norm_by_words"],
                    step=uidx,
                )

            train_loss += losses["Loss"] / y.size(1)
    except RuntimeError as e:
        if "out of memory" in str(e):
            WARN("| WARNING: ran out of memory, skipping batch")
            is_oom = True
            optim.zero_grad()
        else:
            raise

    losses["train_loss"] = train_loss
    return {
        "train_losses": losses,
        "n_samples_t": n_samples_t,
        "n_words_t": n_words_t,
        "is_oom": is_oom,
    }


def loss_validation(model, valid_iterator, step=0):
    n_sents = 0
    n_tokens = 0.0
    sum_loss = defaultdict(float)

    valid_iter = valid_iterator.build_generator()

    for batch in valid_iter:
        _, seqs_x, seqs_y = batch

        n_sents += len(seqs_x)
        n_tokens += sum(len(s) for s in seqs_y)

        x, y = prepare_data(seqs_x, seqs_y, cuda=GlobalNames.USE_GPU)

        loss = compute_forward(model=model, seqs_x=x, seqs_y=y, eval=True, step=step)

        for kk, vv in loss.items():
            sum_loss[kk] += loss[kk]
            if np.any(np.isnan(loss[kk])):
                WARN("NaN detected!")

    for k in sum_loss:
        sum_loss[k] = float("%.2f" % (sum_loss[k] / n_sents))

    return sum_loss


def bleu_validation(
    uidx,
    valid_iterator,
    model,
    bleu_scorer,
    vocab_tgt,
    batch_size,
    valid_dir="./valid",
    max_steps=10,
    beam_size=5,
    alpha=-1.0,
    beta=0.1,
    gamma=0.0,
    reranking=False,
    is_forward=True,
):
    model.eval()

    if is_forward:
        src_lang, tgt_lang = "src", "tgt"
    else:
        src_lang, tgt_lang = "tgt", "src"

    numbers = []
    trans = []

    infer_progress_bar = tqdm(
        total=len(valid_iterator),
        desc=" - (Infer {})  ".format("X->Y" if is_forward else "Y->X"
        ),
        unit="sents",
    )

    valid_iter = valid_iterator.build_generator(batch_size=batch_size)

    for batch in valid_iter:

        seq_nums = batch[0]
        numbers += seq_nums

        seqs_x = batch[1] if is_forward else batch[2]

        infer_progress_bar.update(len(seqs_x))

        x = prepare_data(seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():
            word_ids = mirror_iterative_decoding_v2(
                model,
                src=x,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                iterations=model.args.decoding_iterations,
                reranking=reranking,
                beam_size=beam_size,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != PAD] for line in sent_t]
            x_tokens = []

            for wid in sent_t[0]:
                if wid == EOS:
                    break
                x_tokens.append(vocab_tgt.id2token(wid))

            if len(x_tokens) > 0:
                trans.append(vocab_tgt.tokenizer.detokenize(x_tokens))
            else:
                trans.append("%s" % vocab_tgt.id2token(EOS))

    origin_order = np.argsort(numbers).tolist()
    trans = [trans[ii] for ii in origin_order]

    infer_progress_bar.close()

    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)

    hyp_path = os.path.join(valid_dir, "trans.iter{0}.txt".format(uidx))

    with open(hyp_path, "w") as f:
        for line in trans:
            f.write("%s\n" % line)

    with open(hyp_path) as f:
        bleu_v = bleu_scorer.corpus_bleu(f)

    return bleu_v


def validate_and_update_bleu_stats(
    uidx,
    valid_iterator,
    training_configs,
    model,
    bleu_scorer,
    vocab_tgt,
    valid_path,
    model_collections,
    summary_writer,
    is_forward=True,
):
    suffix = "/forward" if is_forward else "/backward"
    bleu_valid_configs = training_configs["bleu_valid_configs"][0 if is_forward else 1]

    valid_bleu = bleu_validation(
        uidx=uidx,
        valid_iterator=valid_iterator,
        batch_size=training_configs["bleu_valid_batch_size"],
        model=model,
        bleu_scorer=bleu_scorer,
        vocab_tgt=vocab_tgt,
        valid_dir=valid_path + suffix,
        max_steps=bleu_valid_configs["max_steps"],
        beam_size=bleu_valid_configs["beam_size"],
        alpha=bleu_valid_configs["alpha"],
        beta=bleu_valid_configs["beta"],
        gamma=bleu_valid_configs["gamma"],
        reranking=bleu_valid_configs["reranking"],
        is_forward=is_forward,
    )

    model_collections.add_to_collection(key="history_bleus" + suffix, value=valid_bleu)

    best_valid_bleu = float(
        np.array(model_collections.get_collection("history_bleus" + suffix)).max()
    )

    summary_writer.add_scalar("bleu" + suffix, valid_bleu, uidx)
    summary_writer.add_scalar("best_bleu" + suffix, best_valid_bleu, uidx)

    if is_forward:
        summary_writer.add_scalar("bleu", valid_bleu, uidx)
        summary_writer.add_scalar("best_bleu", best_valid_bleu, uidx)

    return valid_bleu, best_valid_bleu


def load_pretrained_model(nmt_model, pretrain_path, device, exclude_prefix=None):
    """
    Args:
        nmt_model: model.
        pretrain_path ('str'): path to pretrained model.
        map_dict ('dict'): mapping specific parameter names to those names
            in current model.
        exclude_prefix ('dict'): excluding parameters with specific names
            for pretraining.

    Raises:
        ValueError: Size not match, parameter name not match or others.

    """
    if exclude_prefix is None:
        exclude_prefix = []
    if pretrain_path != "":
        INFO("Loading pretrained model from {}".format(pretrain_path))
        pretrain_params = torch.load(pretrain_path, map_location=device)
        for name, params in pretrain_params.items():
            flag = False
            for pp in exclude_prefix:
                if name.startswith(pp):
                    flag = True
                    break
            if flag:
                continue
            INFO("Loading param: {}...".format(name))
            try:
                nmt_model.load_state_dict({name: params}, strict=False)
            except Exception as e:
                WARN("{}: {}".format(str(Exception), e))

        INFO("Pretrained model loaded.")


def write_to_tensorboard(
    stats: {}, prefix: str, writer: SummaryWriter, global_step: int
) -> None:
    for stat_k, stat_v in stats.items():
        writer.add_scalar(
            "{}/{}".format(prefix, stat_k), stat_v, global_step=global_step
        )


def change(m):
    return {"src": "tgt", "tgt": "src"}[m]


def train(FLAGS, init_distributed=False):
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(
        os.path.join(FLAGS.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S"))
    )

    GlobalNames.USE_GPU = FLAGS.use_gpu

    if GlobalNames.USE_GPU:
        CURRENT_DEVICE = "cpu"
    else:
        CURRENT_DEVICE = "cuda"
        # CURRENT_DEVICE = FLAGS.device_id

    # [dist] Initialize CUDA and distributed training
    if torch.cuda.is_available():
        FLAGS.device = "cuda"
    else:
        FLAGS.device = "cpu"

    if init_distributed:
        torch.cuda.set_device(FLAGS.device_id)
        FLAGS.distributed_rank = distributed_utils.distributed_init(FLAGS)
        GlobalNames.IS_MASTER = distributed_utils.is_master(FLAGS)
        GlobalNames.DIST_RANK = FLAGS.distributed_rank
        # torch.distributed.init_process_group(
        #     backend='nccl',
        #     init_method=FLAGS.distributed_init_method,
        #     world_size=FLAGS.distributed_world_size,
        #     rank=FLAGS.distributed_rank
        # )
        # if distributed_utils.is_master(FLAGS):
        #     logging.getLogger(__name__).setLevel(logging.INFO)
        # else:
        #     logging.getLogger(__name__).setLevel(logging.WARN)

    config_path = os.path.abspath(FLAGS.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    INFO(pretty_configs(configs))

    # Add default configs
    configs = default_configs(configs)
    data_configs = configs["data_configs"]
    model_configs = configs["model_configs"]
    optimizer_configs = configs["optimizer_configs"]
    training_configs = configs["training_configs"]

    GlobalNames.SEED = training_configs["seed"]

    set_seed(GlobalNames.SEED)

    best_model_prefix = os.path.join(
        FLAGS.saveto, FLAGS.model_name + GlobalNames.MY_BEST_MODEL_SUFFIX
    )

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO("Loading data...")
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    train_batch_size = training_configs["batch_size"] * max(
        1, training_configs["update_cycle"]
    )
    train_buffer_size = training_configs["buffer_size"] * max(
        1, training_configs["update_cycle"]
    )

    train_bitext_dataset = ZipDataset(
        TextLineDataset(
            data_path=data_configs["train_data"][0],
            vocabulary=vocab_src,
            max_len=data_configs["max_len"][0],
        ),
        TextLineDataset(
            data_path=data_configs["train_data"][1],
            vocabulary=vocab_tgt,
            max_len=data_configs["max_len"][1],
        ),
        # shuffle=training_configs['shuffle']
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs["valid_data"][0], vocabulary=vocab_src),
        TextLineDataset(data_path=data_configs["valid_data"][1], vocabulary=vocab_tgt),
    )

    training_iterator = DataIterator(
        dataset=train_bitext_dataset,
        batch_size=train_batch_size,
        use_bucket=training_configs["use_bucket"],
        buffer_size=train_buffer_size,
        batching_func=training_configs["batching_key"],
        world_size=FLAGS.distributed_world_size if init_distributed else 1,
        rank=FLAGS.distributed_rank if init_distributed else 0,
    )

    valid_iterator = DataIterator(
        dataset=valid_bitext_dataset,
        batch_size=training_configs["valid_batch_size"],
        use_bucket=True,
        buffer_size=100000,
        numbering=True,
        batching_func=training_configs["batching_key"],
    )

    enable_NP = training_configs["train_nonparallel_options"]["enable"]
    if enable_NP:
        # nonparallel dataset
        train_monotext_dataset = {
            "src": ZipDataset(
                TextLineDataset(
                    data_path=data_configs["train_data_mono"]["src"],
                    vocabulary=vocab_src,
                    max_len=data_configs["max_len"][0],
                ),
                # shuffle=training_configs['shuffle']
            ),
            "tgt": ZipDataset(
                TextLineDataset(
                    data_path=data_configs["train_data_mono"]["tgt"],
                    vocabulary=vocab_tgt,
                    max_len=data_configs["max_len"][1],
                )
            )
            # shuffle=training_configs['shuffle']),
        }

        training_iterator_mono = {
            "src": DataIterator(
                dataset=train_monotext_dataset["src"],
                batch_size=train_batch_size,
                use_bucket=training_configs["use_bucket"],
                buffer_size=train_buffer_size,
                batching_func=training_configs["batching_key"],
                world_size=FLAGS.distributed_world_size if init_distributed else 1,
                rank=FLAGS.distributed_rank if init_distributed else 0,
            ),
            "tgt": DataIterator(
                dataset=train_monotext_dataset["tgt"],
                batch_size=train_batch_size,
                use_bucket=training_configs["use_bucket"],
                buffer_size=train_buffer_size,
                batching_func=training_configs["batching_key"],
                world_size=FLAGS.distributed_world_size if init_distributed else 1,
                rank=FLAGS.distributed_rank if init_distributed else 0,
            ),
        }

    bleu_scorer = {
        "forward": SacreBLEUScorer(
            reference_path=data_configs["bleu_valid_reference"][0],
            num_refs=data_configs["num_refs"][0],
            lang_pair=data_configs["lang_pair"][0],
            sacrebleu_args=training_configs["bleu_valid_configs"][0]["sacrebleu_args"],
            postprocess=training_configs["bleu_valid_configs"][0]["postprocess"],
        ),
        "backward": SacreBLEUScorer(
            reference_path=data_configs["bleu_valid_reference"][1],
            num_refs=data_configs["num_refs"][1],
            lang_pair=data_configs["lang_pair"][1],
            sacrebleu_args=training_configs["bleu_valid_configs"][1]["sacrebleu_args"],
            postprocess=training_configs["bleu_valid_configs"][1]["postprocess"],
        ),
    }

    INFO("Done. Elapsed time {0}".format(timer.toc()))

    lrate = optimizer_configs["learning_rate"]
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial
    model_collections = Collections()
    checkpoint_saver = Saver(
        save_prefix="{0}.ckpt".format(os.path.join(FLAGS.saveto, FLAGS.model_name)),
        num_max_keeping=training_configs["num_kept_checkpoints"],
    )

    best_model_saver = BestKSaver(
        save_prefix="{0}.best".format(os.path.join(FLAGS.saveto, FLAGS.model_name)),
        num_max_keeping=training_configs["num_kept_best_checkpoints"],
    )

    # 1. Build Model & Criterion
    INFO("Building model...")
    timer.tic()

    model_args = dict_to_args(
        n_src_vocab=vocab_src.max_n_words,
        n_tgt_vocab=vocab_tgt.max_n_words,
        **model_configs
    )
    nmt_model = build_model(model=model_args.model, args=model_args)
    INFO(nmt_model)

    raw_params, params_total, params_with_embedding = calculate_parameter_stats(
        nmt_model
    )
    INFO("Total raw parameters: {}".format(raw_params))
    INFO("Total parameters: {}".format(params_total))
    INFO(
        "Total parameters (excluding word embeddings): {}".format(params_with_embedding)
    )

    if GlobalNames.IS_MASTER:
        with open(
            os.path.join(FLAGS.saveto, FLAGS.model_name + ".model_details"), "w"
        ) as f:
            print("Model Architecture:", file=f)
            print(nmt_model, file=f)
            print("Total raw parameters: {}".format(raw_params), file=f)
            print("Total parameters: {}".format(params_total), file=f)
            print(
                "Total parameters (excluding word embeddings): {}".format(
                    params_with_embedding
                ),
                file=f,
            )

    # 3. Load pretrained model if needed
    load_pretrained_model(
        nmt_model, FLAGS.pretrain_path, exclude_prefix=None, device="cpu"
    )

    # 2. Move to GPU
    if GlobalNames.USE_GPU:
        nmt_model = nmt_model.cuda()

    # # 3. Load pretrained model if needed
    # load_pretrained_model(nmt_model, FLAGS.pretrain_path, exclude_prefix=None,
    #                       device=CURRENT_DEVICE)

    # 4. Build optimizer
    INFO("Building Optimizer...")
    optim = Optimizer(
        name=optimizer_configs["optimizer"],
        model=nmt_model,
        lr=lrate,
        grad_clip=optimizer_configs["grad_clip"],
        optim_args=optimizer_configs["optimizer_params"],
    )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs["schedule_method"] is not None:
        if optimizer_configs["schedule_method"] == "loss":
            scheduler = ReduceOnPlateauScheduler(
                optimizer=optim, **optimizer_configs["scheduler_configs"]
            )
        elif optimizer_configs["schedule_method"] == "noam":
            scheduler = NoamScheduler(
                optimizer=optim, **optimizer_configs["scheduler_configs"]
            )
        elif optimizer_configs["schedule_method"] == "rsqrt":
            scheduler = RsqrtScheduler(
                optimizer=optim, **optimizer_configs["scheduler_configs"]
            )
        else:
            WARN(
                "Unknown scheduler name {0}. Do not use lr_scheduling.".format(
                    optimizer_configs["schedule_method"]
                )
            )
            scheduler = None
    else:
        scheduler = None

    # 6. build moving average

    if training_configs["moving_average_method"] is not None:
        ma = MovingAverage(
            moving_average_method=training_configs["moving_average_method"],
            named_params=nmt_model.named_parameters(),
            alpha=training_configs["moving_average_alpha"],
        )
    else:
        ma = None

    INFO("Done. Elapsed time {0}".format(timer.toc()))

    # Reload from latest checkpoint
    if FLAGS.reload:
        checkpoint_saver.load_latest(
            model=nmt_model,
            optim=optim,
            lr_scheduler=scheduler,
            collections=model_collections,
            ma=ma,
        )

    # [dist]
    if init_distributed:
        nmt_model = distributed_utils.DistributedModel(FLAGS, nmt_model)

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    cum_samples = 0
    cum_words = 0
    valid_loss = best_valid_loss = float("inf")  # Max Float
    saving_files = []

    # use best model for back-translation or not
    if (
        training_configs["train_nonparallel_options"]["enable"]
        and training_configs["train_nonparallel_options"]["best_model_BT"]
    ):
        INFO("Using best model for BT...")
        best_model = deepcopy(nmt_model)
        best_model_saver.load_latest(model=best_model)
        best_model.eval()
    else:
        best_model = None

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    # Build iterator and progress bar
    training_iter = training_iterator.build_generator()
    training_progress_bar = tqdm(
        desc=" - (Epc {}, Upd {}) ".format(eidx, uidx),
        total=len(training_iterator),
        unit="sents",
    )

    if enable_NP:
        training_iter_mono = {
            "src": training_iterator_mono["src"].build_generator(),
            "tgt": training_iterator_mono["tgt"].build_generator(),
        }

        training_progress_bar_NP = tqdm(
            desc=" - (Epc {}, Upd {}) ".format(eidx, uidx),
            total=len(training_iterator_mono["src"]),
            unit="sents",
        )

    summary_writer = SummaryWriter(log_dir=FLAGS.log_path) if GlobalNames.IS_MASTER else None
    if summary_writer is not None: 
        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

    INFO("Begin training...")

    while True:
        uidx += 1

        if (
            optimizer_configs["schedule_method"] is not None
            and optimizer_configs["schedule_method"] != "loss"
        ):
            scheduler.step(global_step=uidx)

        if (
            training_configs["train_nonparallel_options"]["enable"]
            and training_configs["train_nonparallel_options"]["NP_train_mode"] == "finetune"
            and uidx > training_configs["train_nonparallel_options"]["NP_warmup_step"]
        ):
            pass
        else:
            try:
                bitext_batch = next(training_iter)
            except StopIteration:
                training_iter = training_iterator.build_generator()
                bitext_batch = next(training_iter)
                eidx += 1
                training_progress_bar.reset()
                if summary_writer is not None: 
                    summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

            # training step
            # training with parallel data
            bitext_train_spec = train_step(
                nmt_model, optim, bitext_batch, training_configs, uidx
            )
            train_losses = bitext_train_spec["train_losses"]
            n_samples_t = bitext_train_spec["n_samples_t"]
            n_words_t = bitext_train_spec["n_words_t"]
            is_oom = bitext_train_spec["is_oom"]

            train_loss = train_losses["train_loss"]
            cum_samples += n_samples_t
            cum_words += n_words_t
            oom_count += int(is_oom)

            # ======================
            if (
                ma is not None
                and eidx >= training_configs["moving_average_start_epoch"]
            ):
                ma.step()

            if GlobalNames.IS_MASTER:
                training_progress_bar.set_description(
                    " - (Epc {}, Upd {}, {}) ".format(eidx, uidx, "Parallel"),
                    refresh=False
                )
                training_progress_bar.set_postfix_str(
                    "TrainLoss: {:.2f}, Detail: {}".format(
                        train_loss,
                        get_tensor_dict_str(**train_losses),
                    ),
                    refresh=False
                )
                training_progress_bar.update(n_samples_t)
                if summary_writer is not None: 
                    summary_writer.add_scalar(
                        "train_loss", scalar_value=train_loss, global_step=uidx
                    )

        # ============
        # Training with BT data
        # =====
        if (
            training_configs["train_nonparallel_options"]["enable"]
            and uidx > training_configs["train_nonparallel_options"]["NP_warmup_step"]
        ) or FLAGS.debug:
            cycle_oom = 0
            optim.zero_grad()
            for src_lang, tgt_lang in [("src", "tgt"), ("tgt", "src")]:
                try:
                    monotext_batch = next(training_iter_mono[tgt_lang])
                except StopIteration:
                    training_iter_mono[tgt_lang] = training_iterator_mono[
                        tgt_lang
                    ].build_generator()
                    monotext_batch = next(training_iter_mono[tgt_lang])

                    if src_lang == "src":
                        eidx += 1
                        training_progress_bar_NP.reset()
                        if summary_writer is not None: 
                            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

                # training step
                monotext_train_spec = train_step_mono(
                    nmt_model,
                    best_model=best_model,
                    optim=optim,
                    tgt_batch=monotext_batch,
                    training_configs=training_configs,
                    uidx=uidx,
                    src_vocab={"src": vocab_src, "tgt": vocab_tgt}[src_lang],
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                )

                train_losses = monotext_train_spec["train_losses"]
                n_samples_t = monotext_train_spec["n_samples_t"]
                n_words_t = monotext_train_spec["n_words_t"]
                is_oom = monotext_train_spec["is_oom"]

                train_loss = train_losses["train_loss"]
                cum_samples += n_samples_t
                cum_words += n_words_t
                oom_count += int(is_oom)
                cycle_oom += int(is_oom)

                training_progress_bar_NP.set_description(
                    " - (Epc {}, Upd {}, NP:{}->{}) ".format(
                        eidx, uidx, src_lang, tgt_lang
                    ),
                    refresh=False
                )
                training_progress_bar_NP.set_postfix_str(
                    "TrainLoss: {:.2f}, Detail: {}".format(
                        train_loss,
                        get_tensor_dict_str(**train_losses),
                    ),
                    refresh=False
                )
                training_progress_bar_NP.update(n_samples_t)
                if summary_writer is not None:
                    summary_writer.add_scalar(
                        "train_loss/{}2{}".format(src_lang, tgt_lang),
                        scalar_value=train_loss,
                        global_step=uidx,
                    )

            # Update parameters when gradients of forward and backward models have been jointly computed
            if cycle_oom == 0:
                optim.step()

        # ================================================================================== #
        # Display some information
        if should_trigger_by_steps(
            uidx, eidx, every_n_step=training_configs["disp_freq"]
        ):
            # words per second and sents per second
            words_per_sec = cum_words / (timer.toc(return_seconds=True))
            sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
            lrate = list(optim.get_lrate())[0]

            summary_writer.add_scalar(
                "Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx
            )
            summary_writer.add_scalar(
                "Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx
            )
            summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
            summary_writer.add_scalar(
                "oom_count", scalar_value=oom_count, global_step=uidx
            )

            # Reset timer
            timer.tic()
            cum_words = 0
            cum_samples = 0

        # ================================================================================== #
        # Loss Validation & Learning rate annealing
        if should_trigger_by_steps(
            global_step=uidx,
            n_epoch=eidx,
            every_n_step=training_configs["loss_valid_freq"],
            debug=FLAGS.debug,
        ):

            if ma is not None:
                origin_state_dict = deepcopy(nmt_model.state_dict())
                nmt_model.load_state_dict(ma.export_ma_params(), strict=False)

            valid_losses = loss_validation(
                model=nmt_model, valid_iterator=valid_iterator, step=uidx
            )

            valid_loss = valid_losses.pop("Loss")
            print_tensor_dict(uidx=uidx, **valid_losses)
            with open(os.path.join(FLAGS.saveto, "valid.txt"), "a") as f:
                print(get_tensor_dict_str(uidx=uidx, **valid_losses), file=f)

            model_collections.add_to_collection("history_losses", valid_loss)

            min_history_loss = np.array(
                model_collections.get_collection("history_losses")
            ).min()

            summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
            summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)

            best_valid_loss = min_history_loss

            # logging stats to Tensorboard
            valid_losses["kl_weight"] = nmt_model.kl_weight
            write_to_tensorboard(
                stats=valid_losses,
                prefix="loss",
                writer=summary_writer,
                global_step=uidx,
            )

            if ma is not None:
                nmt_model.load_state_dict(origin_state_dict)
                del origin_state_dict

            if optimizer_configs["schedule_method"] == "loss":
                scheduler.step(global_step=uidx, metric=valid_losses["ELBO"])

        # ================================================================================== #
        # BLEU Validation & Early Stop

        if should_trigger_by_steps(
            global_step=uidx,
            n_epoch=eidx,
            every_n_step=training_configs["bleu_valid_freq"],
            min_step=training_configs["bleu_valid_warmup"],
            debug=FLAGS.debug,
        ):

            if ma is not None:
                origin_state_dict = deepcopy(nmt_model.state_dict())
                nmt_model.load_state_dict(ma.export_ma_params(), strict=False)

            # forward validation
            fwd_bleu, fwd_best_bleu = validate_and_update_bleu_stats(
                uidx,
                valid_iterator,
                training_configs,
                nmt_model,
                bleu_scorer["forward"],
                vocab_tgt,
                FLAGS.valid_path,
                model_collections,
                summary_writer,
                is_forward=True,
            )

            # backward validation
            bwd_bleu, bwd_best_bleu = validate_and_update_bleu_stats(
                uidx,
                valid_iterator,
                training_configs,
                nmt_model,
                bleu_scorer["backward"],
                vocab_src,
                FLAGS.valid_path,
                model_collections,
                summary_writer,
                is_forward=False,
            )
            torch.cuda.empty_cache()

            valid_bleu = fwd_bleu + bwd_bleu
            best_valid_bleu = fwd_best_bleu + bwd_best_bleu

            # write local log
            valid_losses = loss_validation(
                model=nmt_model, valid_iterator=valid_iterator, step=uidx
            )

            printed_info = dict(
                uidx=uidx,
                fwd_bleu=fwd_bleu,
                fwd_best_bleu=fwd_best_bleu,
                bwd_bleu=bwd_bleu,
                bwd_best_bleu=bwd_best_bleu,
                **valid_losses
            )

            print_tensor_dict(**printed_info)
            with open(os.path.join(FLAGS.saveto, "valid.txt"), "a") as f:
                print(get_tensor_dict_str(**printed_info), file=f)

            # If model get new best valid bleu score
            if valid_bleu >= best_valid_bleu:
                bad_count = 0

                # update best model for better BT
                if (
                    training_configs["train_nonparallel_options"]["enable"]
                    and training_configs["train_nonparallel_options"]["best_model_BT"]
                ):
                    best_model.load_state_dict(
                        nmt_model.state_dict()
                        if not hasattr(nmt_model, "module")
                        else nmt_model.module.state_dict()
                    )
                    best_model.eval()

                if True or is_early_stop is False:
                    # 1. save the best model's parameters
                    torch.save(
                        nmt_model.state_dict()
                        if not hasattr(nmt_model, "module")
                        else nmt_model.module.state_dict(),
                        best_model_prefix + ".final",
                    )

                    # 2. save the best checkpoint

                    model_collections.add_to_collection("uidx", uidx)
                    model_collections.add_to_collection("eidx", eidx)
                    model_collections.add_to_collection("bad_count", bad_count)

                    best_model_saver.save(
                        global_step=uidx,
                        metric=valid_bleu,
                        model=nmt_model,
                        optim=optim,
                        lr_scheduler=scheduler,
                        collections=model_collections,
                        ma=ma,
                    )
            else:
                bad_count += 1

                # At least one epoch should be traversed
                if bad_count >= training_configs["early_stop_patience"] and eidx > 0:
                    is_early_stop = True
                    WARN("Early Stop!")

            summary_writer.add_scalar("bad_count", bad_count, uidx)

            # save best forward and backward model.
            if fwd_bleu >= fwd_best_bleu:
                torch.save(
                    nmt_model.state_dict()
                    if not hasattr(nmt_model, "module")
                    else nmt_model.module.state_dict(),
                    best_model_prefix + "_fwd.final",
                )

            if bwd_bleu >= bwd_best_bleu:
                torch.save(
                    nmt_model.state_dict()
                    if not hasattr(nmt_model, "module")
                    else nmt_model.module.state_dict(),
                    best_model_prefix + "_bwd.final",
                )


            if ma is not None:
                nmt_model.load_state_dict(origin_state_dict)
                del origin_state_dict

            INFO(
                "{0} Loss: {1:.2f} BLEU: {2:.2f} lrate: {3:6f} patience: {4}".format(
                    uidx, valid_loss, valid_bleu, lrate, bad_count
                )
            )

        # ================================================================================== #
        # Saving checkpoints
        if GlobalNames.IS_MASTER and should_trigger_by_steps(
            uidx, eidx, every_n_step=training_configs["save_freq"], debug=FLAGS.debug
        ):
            model_collections.add_to_collection("uidx", uidx)
            model_collections.add_to_collection("eidx", eidx)
            model_collections.add_to_collection("bad_count", bad_count)

            if not is_early_stop:
                checkpoint_saver.save(
                    global_step=uidx,
                    model=nmt_model,
                    optim=optim,
                    lr_scheduler=scheduler,
                    collections=model_collections,
                    ma=ma,
                )

        if eidx > training_configs["max_epochs"]:
            training_progress_bar.close()
            break


def translate(FLAGS):
    GlobalNames.USE_GPU = FLAGS.use_gpu

    config_path = os.path.abspath(FLAGS.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    data_configs = configs["data_configs"]
    model_configs = configs["model_configs"]

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO("Loading data...")
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary(**data_configs["vocabularies"][0])
    vocab_tgt = Vocabulary(**data_configs["vocabularies"][1])

    valid_dataset = TextLineDataset(data_path=FLAGS.source_path, vocabulary=vocab_src)

    valid_iterator = DataIterator(
        dataset=valid_dataset,
        batch_size=FLAGS.batch_size,
        use_bucket=True,
        buffer_size=100000,
        numbering=True,
        batching_func="tokens",
    )

    INFO("Done. Elapsed time {0}".format(timer.toc()))

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO("Building model...")
    timer.tic()
    model_args = dict_to_args(
        n_src_vocab=vocab_src.max_n_words,
        n_tgt_vocab=vocab_tgt.max_n_words,
        **model_configs
    )
    model = build_model(model=model_args.model, args=model_args)

    # model = build_model(n_src_vocab=vocab_src.max_n_words,
    #                         n_tgt_vocab=vocab_tgt.max_n_words, **model_configs)
    model.eval()
    INFO("Done. Elapsed time {0}".format(timer.toc()))

    INFO("Reloading model parameters...")
    timer.tic()

    params = load_model_parameters(FLAGS.model_path, map_location="cpu")

    model.load_state_dict(params)

    if GlobalNames.USE_GPU:
        model.cuda()

    INFO("Done. Elapsed time {0}".format(timer.toc()))

    INFO("Begin...")

    result_numbers = []
    result = []
    n_words = 0

    timer.tic()

    infer_progress_bar = tqdm(
        total=len(valid_iterator),
        desc=" - (Infer) {}->{} ".format(FLAGS.src_lang, change(FLAGS.src_lang)),
        unit="sents",
    )

    valid_iter = valid_iterator.build_generator()

    if FLAGS.src_lang == "tgt":
        vocab_src, vocab_tgt = vocab_tgt, vocab_src

    for batch in valid_iter:

        numbers, seqs_x = batch

        batch_size_t = len(seqs_x)

        x = prepare_data(seqs_x=seqs_x, cuda=GlobalNames.USE_GPU)

        with torch.no_grad():

            word_ids = mirror_iterative_decoding_v2(
                model,
                src=x,
                src_lang=FLAGS.src_lang,
                tgt_lang=change(FLAGS.src_lang),
                iterations=model.args.decoding_iterations,
                beam_size=FLAGS.beam_size,
                alpha=FLAGS.alpha,
                reranking=FLAGS.reranking,
                beta=FLAGS.beta,
                gamma=FLAGS.gamma,
            )

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != PAD] for line in sent_t]
            result.append(sent_t)

            n_words += len(sent_t[0])

        result_numbers += numbers

        infer_progress_bar.update(batch_size_t)

    infer_progress_bar.close()

    INFO(
        "Done. Speed: {0:.2f} words/sec".format(
            n_words / (timer.toc(return_seconds=True))
        )
    )

    translation = []
    for sent in result:
        samples = []
        for trans in sent:
            sample = []
            for w in trans:
                if w == vocab_tgt.EOS:
                    break
                sample.append(vocab_tgt.id2token(w))
            samples.append(vocab_tgt.tokenizer.detokenize(sample))
        translation.append(samples)

    # resume the ordering
    origin_order = np.argsort(result_numbers).tolist()
    translation = [translation[ii] for ii in origin_order]

    keep_n = (
        FLAGS.beam_size if FLAGS.keep_n <= 0 else min(FLAGS.beam_size, FLAGS.keep_n)
    )
    outputs = ["%s.%d" % (FLAGS.saveto, i) for i in range(keep_n)]

    with batch_open(outputs, "w") as handles:
        for trans in translation:
            for i in range(keep_n):
                if i < len(trans):
                    handles[i].write("%s\n" % trans[i])
                else:
                    handles[i].write("%s\n" % "eos")


if __name__ == "__main__":
    _args = {
        "model_name": "test_cgnmt",
        "reload": False,
        "config_path": "./configs/test_cgnmt_iter.yaml",
        "debug": True,
        "use_gpu": False,
        "task": "cgnmt_iter",
        "log_path": "/tmp",
        "saveto": "/tmp",
        "valid_path": "/tmp",
    }

    from src.bin import train as _train

    _train.run(**_args)
