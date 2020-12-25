import os
import torch
import time
import contextlib
import copy
import numpy as np

from . import nest
# from src.utils.logging import INFO, WARN

__all__ = [
    'batch_open',
    'GlobalNames',
    'Timer',
    'Collections',
    'build_vocab_shortlist',
    'to_gpu',
    'should_trigger_by_steps',
    'Saver',
    'BestKSaver'
]


def argsort(seq, reverse=False):
    numbered_seq = [(i, e) for i, e in enumerate(seq)]
    sorted_seq = sorted(numbered_seq, key=lambda p: p[1], reverse=reverse)

    return [p[0] for p in sorted_seq]


# ================================================================================== #
# File I/O Utils

@contextlib.contextmanager
def batch_open(refs, mode='r'):
    handlers = []
    if not isinstance(refs, (list, tuple)):
        refs = [refs]
    for f in refs:
        handlers.append(open(f, mode))

    yield handlers

    for h in handlers:
        h.close()


class GlobalNames:
    # learning rate variable name
    MY_LEARNING_RATE_NAME = "learning_rate"

    MY_CHECKPOINIS_PREFIX = ".ckpt"

    MY_BEST_MODEL_SUFFIX = ".best"

    MY_COLLECTIONS_SUFFIX = ".collections.pkl"

    MY_MODEL_ARCHIVES_SUFFIX = ".archives.pkl"

    USE_GPU = False

    SEED = 314159
    
    IS_MASTER = True
    DIST_RANK = 0


time_format = '%Y-%m-%d %H:%M:%S'


class Timer(object):
    def __init__(self):
        self.t0 = 0

    def tic(self):
        self.t0 = time.time()

    def toc(self, format='m:s', return_seconds=False):
        t1 = time.time()

        if return_seconds is True:
            return t1 - self.t0

        if format == 's':
            return '{0:d}'.format(t1 - self.t0)
        m, s = divmod(t1 - self.t0, 60)
        if format == 'm:s':
            return '%d:%02d' % (m, s)
        h, m = divmod(m, 60)
        return '%d:%02d:%02d' % (h, m, s)


class Collections(object):
    """Collections for logs during training.

    Usually we add loss and valid metrics to some collections after some steps.
    """
    _MY_COLLECTIONS_NAME = "my_collections"

    def __init__(self, kv_stores=None, name=None):

        self._kv_stores = kv_stores if kv_stores is not None else {}

        if name is None:
            name = Collections._MY_COLLECTIONS_NAME
        self._name = name

    def add_to_collection(self, key, value):
        """
        Add value to collection

        :type key: str
        :param key: Key of the collection

        :param value: The value which is appended to the collection
        """
        if key not in self._kv_stores:
            self._kv_stores[key] = [value]
        else:
            self._kv_stores[key].append(value)

    def get_collection(self, key, default=[]):
        """
        Get the collection given a key

        :type key: str
        :param key: Key of the collection
        """
        if key not in self._kv_stores:
            return default
        else:
            return self._kv_stores[key]

    def state_dict(self):

        return self._kv_stores

    def load_state_dict(self, state_dict):

        self._kv_stores = copy.deepcopy(state_dict)


def build_vocab_shortlist(shortlist):
    shortlist_ = nest.flatten(shortlist)

    shortlist_ = sorted(list(set(shortlist_)))

    shortlist_np = np.array(shortlist_).astype('int64')

    map_to_shortlist = dict([(wid, sid) for sid, wid in enumerate(shortlist_np)])
    map_from_shortlist = dict([(item[1], item[0]) for item in map_to_shortlist.items()])

    return shortlist_np, map_to_shortlist, map_from_shortlist


def to_gpu(*inputs):
    return list(map(lambda x: x.cuda(), inputs))


def _min_cond_to_trigger(global_step, n_epoch, min_step=-1):
    """
    If min_step is an integer within (0,10]

    global_step is the minimum number of epochs to trigger action.
    Otherwise it is the minimum number of steps.
    """
    if min_step > 0 and min_step <= 50:
        if n_epoch >= min_step:
            return True
        else:
            return False
    else:
        if global_step >= min_step:
            return True
        else:
            return False


def should_trigger_by_steps(global_step,
                            n_epoch,
                            every_n_step,
                            min_step=-1,
                            debug=False):
    """
    When to trigger bleu evaluation.
    """
    # Debug mode

    if not GlobalNames.IS_MASTER:
        return False

    if debug:
        return True

    # Not setting condition

    if every_n_step <= 0:
        return False

    if _min_cond_to_trigger(global_step=global_step, n_epoch=n_epoch, min_step=min_step):

        if np.mod(global_step, every_n_step) == 0:
            return True
        else:
            return False


class Saver(object):
    """ Saver to save and restore objects.

    Saver only accept objects which contain two method: ```state_dict``` and ```load_state_dict```
    """

    def __init__(self, save_prefix, num_max_keeping=1):

        self.save_prefix = save_prefix.rstrip(".")

        save_dir = os.path.dirname(self.save_prefix)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.save_dir = save_dir

        if os.path.exists(self.save_prefix):
            with open(self.save_prefix) as f:
                save_list = f.readlines()
            save_list = [line.strip() for line in save_list]
        else:
            save_list = []

        self.save_list = save_list
        self.num_max_keeping = num_max_keeping

    @staticmethod
    def savable(obj):

        if hasattr(obj, "state_dict") and hasattr(obj, "load_state_dict"):
            return True
        else:
            return False

    def save(self, global_step, **kwargs):

        state_dict = dict()

        for key, obj in kwargs.items():
            if self.savable(obj):
                state_dict[key] = obj.state_dict() if not hasattr(obj, "module") else obj.module.state_dict()

        saveto_path = '{0}.{1}'.format(self.save_prefix, global_step)
        torch.save(state_dict, saveto_path)

        self.save_list.append(os.path.basename(saveto_path))

        if len(self.save_list) > self.num_max_keeping:
            out_of_date_state_dict = self.save_list.pop(0)
            os.remove(os.path.join(self.save_dir, out_of_date_state_dict))

        with open(self.save_prefix, "w") as f:
            f.write("\n".join(self.save_list))

    def load_latest(self, **kwargs):

        if len(self.save_list) == 0:
            return

        latest_path = os.path.join(self.save_dir, self.save_list[-1])
        from torch.serialization import default_restore_location
        state_dict = torch.load(latest_path, map_location=torch.device("cpu"))

        for name, obj in kwargs.items():
            if self.savable(obj):

                if name not in state_dict:
                    if GlobalNames.IS_MASTER:
                        print("Warning: {0} has no content saved!".format(name))
                else:
                    if GlobalNames.IS_MASTER:
                        print("Loading {0}".format(name))
                    if hasattr(obj, "module"):
                        obj.module.load_state_dict(state_dict[name])
                    else:
                        obj.load_state_dict(state_dict[name])


class BestKSaver(object):
    """ Saving best k checkpoints with some metric.

    `BestKSaver` will save checkpoints with largest k values of metric. If two checkpoints have the same
    value of metric, the latest checkpoint will be saved.
    """

    def __init__(self, save_prefix, num_max_keeping=1):

        self.save_prefix = save_prefix.rstrip(".")

        save_dir = os.path.dirname(self.save_prefix)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.save_dir = save_dir

        if os.path.exists(self.save_prefix):
            with open(self.save_prefix) as f:
                save_list = f.readlines()
            save_names = [line.strip().split(",")[0] for line in save_list]
            save_metrics = [float(line.strip().split(",")[1]) for line in save_list]
        else:
            save_names = []
            save_metrics = []

        self.save_names = save_names
        self.save_metrics = save_metrics

        self.num_max_keeping = num_max_keeping

    @property
    def num_saved(self):
        return len(self.save_names)

    @property
    def min_save_metric(self):

        if len(self.save_metrics) == 0:
            return - 1e-5  # pseudo minimum value, no metric can be less than this
        else:
            return min(self.save_metrics)

    @staticmethod
    def savable(obj):
        if hasattr(obj, "state_dict") and hasattr(obj, "load_state_dict"):
            return True
        else:
            return False

    def get_all_ckpt_path(self):
        """Get all the path of checkpoints contains by"""
        return [os.path.join(self.save_dir, path) for path in self.save_names]

    def save(self, global_step, metric, **kwargs):

        # Less than minimum metric value, do not save
        if metric < self.min_save_metric:
            return

        state_dict = dict()

        for key, obj in kwargs.items():
            if self.savable(obj):
                state_dict[key] = obj.state_dict() if not hasattr(obj, "module") else obj.module.state_dict()

        saveto_path = '{0}.{1}'.format(self.save_prefix, global_step)
        torch.save(state_dict, saveto_path)

        if self.num_saved == 0:
            new_save_names = [os.path.basename(saveto_path), ]
            new_save_metrics = [metric, ]
        else:
            # new metric less than i-1 th checkpoint
            insert_pos = 0

            for i in range(len(self.save_metrics)):
                if metric >= self.save_metrics[i]:
                    insert_pos = i
                    break
            new_save_names = self.save_names[:insert_pos] + [os.path.basename(saveto_path), ] \
                             + self.save_names[insert_pos:]
            new_save_metrics = self.save_metrics[:insert_pos] + [metric, ] + self.save_metrics[insert_pos:]

        if len(new_save_names) > self.num_max_keeping:
            out_of_date_name = new_save_names[-1]
            os.remove(os.path.join(self.save_dir, out_of_date_name))
            new_save_names = new_save_names[:-1]
            new_save_metrics = new_save_metrics[:-1]

        self.save_names = new_save_names
        self.save_metrics = new_save_metrics

        with open(self.save_prefix, "w") as f:
            for ii in range(len(self.save_names)):
                f.write("{0},{1}\n".format(self.save_names[ii], self.save_metrics[ii]))

    def load_latest(self, **kwargs):

        if len(self.save_names) == 0:
            return

        latest_path = os.path.join(self.save_dir, self.save_names[-1])

        state_dict = torch.load(latest_path, map_location=torch.device("cpu"))

        for name, obj in kwargs.items():
            if self.savable(obj):

                if name not in state_dict:
                    if GlobalNames.IS_MASTER:
                        print("Warning: {0} has no content saved!".format(name))
                else:
                    if GlobalNames.IS_MASTER:
                        print("Loading {0}".format(name))
                    if hasattr(obj, "module"):
                        obj.module.load_state_dict(state_dict[name])
                    else:
                        obj.load_state_dict(state_dict[name])

    def clean_all_checkpoints(self):

        # remove all the checkpoint files
        for ckpt_path in self.get_all_ckpt_path():
            try:
                os.remove(ckpt_path)
            except:
                continue
        try:
            os.remove(self.save_prefix)
        except:
            pass
        # rest save list
        self.save_names = []
        self.save_metrics = []


def calculate_parameter_stats(model):
    raw_params = sum([p.numel() for n, p in model.state_dict().items()])
    params_total = sum([p.numel() for n, p in model.named_parameters()])
    params_with_embedding = sum(
        [p.numel() for n, p in model.named_parameters() if
         n.find('embedding') == -1])

    return raw_params, params_total, params_with_embedding
