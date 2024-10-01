from __future__ import print_function

from datetime import datetime
import os

import dotenv
import git
import matplotlib
import mlflow
import optuna
import seaborn as sns
from mlflow import MlflowClient

from utilities.utils import save_parameters
from diffusion.runner import run

matplotlib.use('Agg')
# Sns style
sns.set_theme()
sns.set_style("ticks")
sns.set_context('paper')

dotenv.load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))

# Git and MLFlow settings
repo = git.Repo("./")
githash = str(repo.head.object.hexsha)
gitmessage = str(repo.head.object.message)
gitbranch = str(repo.active_branch)
git_remote_url = str(repo.remotes.origin.url)

client = MlflowClient()
# Create an experiment with a unique name
experiment_name = f"exp_{datetime.now().strftime('%y%m%d_%H%M%S')}"
experiment_id = mlflow.create_experiment(f"{experiment_name}")
client.set_experiment_tag(experiment_id, "mlflow.note.content",
                          f"{gitbranch}:{githash[:6]}:{git_remote_url}:{gitmessage}")


def objective(trial):
    run_name = f"Trial_{trial.number}"
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        # Hyperparameters to tune
        batch_size = trial.suggest_int("batch_size", 8, 300)
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        epochs = trial.suggest_int("epochs", 3, 16)
        diffusion_steps = trial.suggest_int("diffusion_steps", 2, 6)

        score = run(experiment_id=experiment_id, batch_size=batch_size, lr=lr, epochs=epochs,
                    diffusion_steps=diffusion_steps)

        return score


study = optuna.create_study(study_name="Wi-Fi-study", direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)
save_parameters(study=study)
