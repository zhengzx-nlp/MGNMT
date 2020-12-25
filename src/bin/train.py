import argparse
import os

from . import auto_mkdir, import_task
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str,
                    help="The name of the model. Will alse be the prefix of saving archives.")

parser.add_argument('--reload', action="store_true",
                    help="Whether to restore from the latest archives.")

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--log_path', type=str, default="./log",
                    help="The path for saving tensorboard logs. Default is ./log")

parser.add_argument('--saveto', type=str, default="./save",
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--debug', action="store_true",
                    help="Use debug mode.")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

parser.add_argument('--pretrain_path', type=str, default="",
                    help="The path for pretrained model.")

parser.add_argument("--valid_path", type=str, default="./valid",
                    help="""Path to save translation for bleu evaulation. Default is ./valid.""")

parser.add_argument("--task", type=str, default="baseline",
                    help="""Choose to run which task.""")


def distributed_run(i, args, train_fn, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    # task.train(args, init_distributed=True)
    train_fn(args, init_distributed=True)
        
def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    auto_mkdir(args.log_path)
    auto_mkdir(args.saveto)
    auto_mkdir(args.valid_path)

    task = import_task(args.task)
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        import random
        port = random.randint(10000, 20000) 
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)

        args.distributed_world_size = torch.cuda.device_count()
        print("world_size = {}".format(args.distributed_world_size))
        args.distributed_rank = None
        torch.multiprocessing.spawn(
            fn=distributed_run,
            args=(args, task.train),
            nprocs=args.distributed_world_size)
    else:
        task.train(args)


if __name__ == '__main__':
    run()
