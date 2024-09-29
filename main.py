from __future__ import print_function

import os

import dotenv
import matplotlib
import mlflow
import optuna
import seaborn as sns

from utilities.utils import save_parameters
from diffusion.runner import run

matplotlib.use('Agg')
# Sns style
sns.set_theme()
sns.set_style("ticks")
sns.set_context('paper')

dotenv.load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))


def objective(trial):
    # Hyperparameters
    batch_size = trial.suggest_int("batch_size", 8, 300)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    epochs = trial.suggest_int("epochs", 3, 16)
    diffusion_steps = trial.suggest_int("diffusion_steps", 2, 6)

    score = run(batch_size=batch_size, lr=lr, epochs=epochs, diffusion_steps=diffusion_steps)
    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)
save_parameters(study=study)
