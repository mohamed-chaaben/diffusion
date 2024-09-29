import os
import sys

import dotenv
import mlflow
from optuna import Study

import os
import random
from pathlib import Path

import git
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from torch import nn
from tqdm import tqdm
from utilities.data_utils import load_and_prep_data
from table_evaluator import TableEvaluator


def save_parameters(study: Study, file_path: str = 'parameters.txt'):
    # Save the best parameters found by Optuna
    with open(file_path, 'a') as file:
        for key, value in study.best_params.items():
            file.write(f"{key}: {value}\n")
    print(f"Parameters are saved to {file_path}")


def configure_directories():
    srcdir = "./"
    datadir = Path("data")
    input_data = datadir / "input_data"
    output_data = datadir / "output_data"

    paths_to_check = [srcdir, datadir, input_data, output_data]

    try:
        for p in paths_to_check:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} does not exist")
        sys.path.append(str(srcdir))

    except FileNotFoundError as e:
        print(e)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return srcdir, datadir, input_data, output_data


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def gather_object_params(obj, prefix="", clip=249):
    return {
        prefix + k: str(v)[:clip] if len(str(v)) > clip else v for k, v in obj.__dict__.items()
    }


def set_random_seed(seed=None):
    if seed is None:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
