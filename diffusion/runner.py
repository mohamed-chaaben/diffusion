from datetime import datetime
from pathlib import Path

import git
import mlflow
import numpy as np
import pandas as pd
import torch
from ctgan.synthesizers import CTGAN
from mlflow.tracking import MlflowClient
from table_evaluator import TableEvaluator
from tqdm import tqdm

from .table_diffusion import TableDiffusion
from utilities.data_utils import load_and_prep_data, datasets
from utilities.metrics import compute_marginal_distances
from utilities.utils import set_random_seed, configure_directories, configure_mlflow

configure_mlflow()
def log_mlflow_info(githash, gitbranch, git_remote_url, gitmessage, repeat, repeats, _seed, metaseed, _seeds, X, _X,
                    disc_cols, data_params, cuda):
    mlflow.set_tags({
        "mlflow.source.git.commit": githash,
        "mlflow.source.git.branch": gitbranch,
        "mlflow.source.git.repoURL": git_remote_url,
        "mlflow.note.content": f"{gitbranch}:{githash[:6]}:{gitmessage}",
        "repeat": f"{repeat}/{repeats}",
        "random_seed.run": _seed,
        "random_seed.meta": metaseed,
        "random_seed.seeds": _seeds,
    })
    mlflow.log_params({
        "gpu_properties": str(torch.cuda.get_device_properties(0)) if cuda else "cpu",
        "dataset.dataset_name": str(data_params),
        "dataset.shape_raw": X.shape,
        "dataset.shape_transformed": _X.shape,
        "dataset.drop_cols": list(data_params["drop_cols"]),
        "dataset.disc_cols": list(disc_cols)
    })


# Unified CTGAN benchmark and fake data generation
def run_ctgan_benchmark(X, disc_cols, dataset, repeat, fake_data_path, ctgan_epochs, cuda):
    ctgan = CTGAN(epochs=ctgan_epochs, cuda=cuda)
    ctgan.fit(train_data=X, discrete_columns=disc_cols)
    X_fake_benchmark = ctgan.sample(X.shape[0])
    pd.DataFrame(X_fake_benchmark).to_csv(Path(fake_data_path) / f"fake_{dataset}_CTGAN_{repeat}.csv", index=False)


# Removed unnecessary parameters and made it simpler
def train_and_evaluate_model(synth, dataset, X, _X, processor, disc_cols, generate_fakes, fake_data_path, repeat):
    # Model params
    model = synth(cuda=True, )
    model.fit(_X.copy(), discrete_columns=disc_cols)

    if generate_fakes:
        X_fake = pd.DataFrame(model.sample(_X.shape[0]))
        X_fake = pd.DataFrame(processor.inverse_transform(X_fake.values), columns=X.columns)

        evaluator = TableEvaluator(X, X_fake, cat_cols=disc_cols, verbose=True)
        score = evaluator.column_correlations()

        marginal_distance = compute_marginal_distances(X, X_fake, [('device_label', 'categorical')])  # Example
        mlflow.log_metrics({f"{key.replace('*', '_')}md": value for key, value in marginal_distance.items()})

        for plot_func, filename in [(evaluator.plot_mean_std, "Log_mea_std.png"),
                                    (evaluator.plot_cumsums, "cumsum.png"),
                                    (evaluator.plot_distributions, "dist.png"),
                                    (evaluator.plot_pca, "pca.png"),
                                    (evaluator.plot_correlation_difference, "corr_difference.png")]:
            mlflow.log_figure(plot_func(), filename)

        mlflow.log_metric("score", score)
        mlflow.log_text(str(X_fake.describe(include="all")), f"fake_stats_{dataset}_TableDiffusion.txt")

        if fake_data_path:
            X_fake.to_csv(Path(fake_data_path) / f"fake_{dataset}_TableDiffusion_{repeat}.csv", index=False)


# Unified run function that replaces run_model and run_synthesiser
def run_experiment(repeats=3, generate_fakes=True, with_benchmark=False, ctgan_epochs=30, cuda=True, metaseed=42):
    # Set up experiment name and directories
    exp_name = f"exp_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    srcdir, datadir, input_data, output_data = configure_directories()
    EXP_PATH = Path(output_data) / exp_name
    FAKE_DSET_PATH = EXP_PATH / "fake_datasets"
    FAKE_DSET_PATH.mkdir(parents=True, exist_ok=True)

    # Check for CUDA availability
    cuda = torch.cuda.is_available() if cuda else False
    print(f"CUDA status: {cuda}")

    # Setup experiment in MLflow
    exp_id = mlflow.create_experiment(exp_name)
    np.random.seed(metaseed)
    _seeds = np.random.randint(10000, size=repeats)

    repo = git.Repo(srcdir)
    githash = str(repo.head.object.hexsha)
    gitmessage = str(repo.head.object.message)
    gitbranch = str(repo.active_branch)
    git_remote_url = str(repo.remotes.origin.url)

    client = MlflowClient()
    client.set_experiment_tag(exp_id, "mlflow.note.content", f"{gitbranch}:{githash[:6]}:{gitmessage}")

    # Run experiment for each repeat
    for repeat in tqdm(range(1, repeats + 1), desc="Repeats", leave=True, colour="red"):
        _seed = _seeds[repeat - 1]
        set_random_seed(_seed)

        for dataset, data_params in datasets.items():
            path = datadir / data_params["path"]
            X, _X, processor = load_and_prep_data(dataset=dataset, datadir=datadir, verbose=False)
            disc_cols = [c for c, dtype in data_params["data_types"] if "categorical" in dtype]
            print(f"Loaded {dataset} dataset {X.shape} from {path}")

            if with_benchmark and FAKE_DSET_PATH:
                run_ctgan_benchmark(X, disc_cols, dataset, repeat, FAKE_DSET_PATH, ctgan_epochs, cuda)

            run_name = f"{exp_name}_{dataset}_TableDiffusion_{repeat}"
            with mlflow.start_run(run_name=run_name, experiment_id=exp_id, nested=True):
                log_mlflow_info(githash, gitbranch, git_remote_url, gitmessage, repeat, repeats, _seed, metaseed,
                                _seeds, X, _X, disc_cols, data_params, cuda)
                train_and_evaluate_model(TableDiffusion, dataset, X, _X, processor, disc_cols, generate_fakes,
                                         FAKE_DSET_PATH, repeat)
