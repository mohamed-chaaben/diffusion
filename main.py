from __future__ import print_function

import os
import sys
from datetime import datetime
from pathlib import Path

import dotenv
import matplotlib
import optuna
import seaborn as sns
from pandas import DataFrame


from diffusion.runner import run_experiment
run_experiment()


