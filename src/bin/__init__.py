import importlib
import os


def auto_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def import_task(task_name):
    return importlib.import_module("src.tasks.{0}".format(task_name))
