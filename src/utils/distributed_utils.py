import logging
import os
import pickle
import socket
import struct
import subprocess
import warnings
from collections import OrderedDict
from typing import Any, Dict, Mapping

import torch
import torch.distributed as dist

# from fairseq import utils


logger = logging.getLogger(__name__)


def is_master(args):
    return args.distributed_rank == 0


def infer_init_method(args):
    if args.distributed_init_method is not None:
        return

    # support torch.distributed.launch
    if all(key in os.environ for key in [
        'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK'
    ]):
        args.distributed_init_method = 'env://'
        args.distributed_world_size = int(os.environ['WORLD_SIZE'])
        args.distributed_rank = int(os.environ['RANK'])

    # we can determine the init method automatically for Slurm
    elif args.distributed_port > 0:
        node_list = os.environ.get('SLURM_STEP_NODELIST')
        if node_list is None:
            node_list = os.environ.get('SLURM_JOB_NODELIST')
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list])
                args.distributed_init_method = 'tcp://{host}:{port}'.format(
                    host=hostnames.split()[0].decode('utf-8'),
                    port=args.distributed_port,
                )
                nnodes = int(os.environ.get('SLURM_NNODES'))
                ntasks_per_node = os.environ.get('SLURM_NTASKS_PER_NODE')
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os.environ.get('SLURM_NTASKS'))
                    nnodes = int(os.environ.get('SLURM_NNODES'))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)
                if ntasks_per_node == 1:
                    assert args.distributed_world_size % nnodes == 0
                    gpus_per_node = args.distributed_world_size // nnodes
                    node_id = int(os.environ.get('SLURM_NODEID'))
                    args.distributed_rank = node_id * gpus_per_node
                else:
                    assert ntasks_per_node == args.distributed_world_size // nnodes
                    args.distributed_no_spawn = True
                    args.distributed_rank = int(os.environ.get('SLURM_PROCID'))
                    args.device_id = int(os.environ.get('SLURM_LOCALID'))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass


def distributed_init(args):
    if args.distributed_world_size == 1:
        raise ValueError('Cannot initialize distributed with distributed_world_size=1')

    if torch.distributed.is_initialized():
        warnings.warn('Distributed is already initialized, cannot initialize twice!')
    else:
        logger.info('distributed init (rank {}): {}'.format(
            args.distributed_rank, args.distributed_init_method,
        ))
        dist.init_process_group(
            backend="nccl",
            init_method=args.distributed_init_method,
            world_size=args.distributed_world_size,
            rank=args.distributed_rank,
        )
        # logger.info('initialized host {} as rank {}'.format(
        #     socket.gethostname(), args.distributed_rank,
        # ))

        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())
        else:
            dist.all_reduce(torch.zeros(1))

        if is_master(args):
            logging.getLogger("NJUNMT").setLevel(logging.INFO)
        else:
            logging.getLogger("NJUNMT").setLevel(logging.WARNING)

    args.distributed_rank = torch.distributed.get_rank()
    return args.distributed_rank


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()



import torch.nn as nn


# inspired by Fairseq
def DistributedModel(args, model):
    """
    Wrap a *model* to support distributed data parallel training.

    This is similar to the built-in DistributedDataParallel, but allows
    additional configuration of the DistributedDataParallel class to
    use, and also provides easier access to the wrapped model by
    forwarding requests for missing attributes to the wrapped model.

    Args:
        args (argparse.Namespace): fairseq args
        model (BaseFairseqModel): model to wrap
    """
    # determine which DDP class to extend
    assert isinstance(model, nn.Module)
    ddp_class = nn.parallel.DistributedDataParallel
    init_kwargs = dict(
        module=model,
        device_ids=[args.device_id],
        output_device=args.device_id,
        # broadcast_buffers=args.broadcast_buffers,
        # bucket_cap_mb=args.bucket_cap_mb,
    )

    class _DistributedModel(ddp_class):
        """Extend DistributedDataParallel to check for missing
        attributes in the wrapped module."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getattr__(self, name):
            wrapped_module = super().__getattr__('module')
            if hasattr(wrapped_module, name):
                return getattr(wrapped_module, name)
            return super().__getattr__(name)

    return _DistributedModel(**init_kwargs)