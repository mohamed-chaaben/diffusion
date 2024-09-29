from pathlib import Path
from datetime import datetime

import git
import mlflow
import pandas as pd
import torch
import numpy as np
from ctgan.synthesizers import CTGAN
from mlflow.tracking import MlflowClient
from table_evaluator import TableEvaluator
from tqdm import tqdm

from utilities.data_utils import load_and_prep_data, dataset
from utilities.metrics import compute_marginal_distances
from utilities.utils import set_random_seed

from .table_diffusion import TableDiffusion


def log_mlflow(run_name, exp_id, githash, gitbranch, git_remote_url, gitmessage, repeat, repeats, _seed, metaseed,
               _seeds, X, _X, discrete_cols, raw_data, cuda):
    with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
        mlflow.set_tags({
            "mlflow.source.git.commit": githash,
            "mlflow.source.git.branch": gitbranch,
            "mlflow.source.git.repoURL": git_remote_url,
            "repeat": f"{repeat}/{repeats}",
            "random_seed.run": _seed,
            "random_seed.meta": metaseed,
            "random_seed.seeds": _seeds,
        })
        mlflow.log_params({
            "gpu_properties": str(torch.cuda.get_device_properties(0)) if cuda else "cpu",
            "dataset.shape_raw": X.shape,
            "dataset.shape_transformed": _X.shape,
            "dataset.raw_data_used": raw_data,
            "dataset.discrete_cols": list(discrete_cols)
        })


def run(model=TableDiffusion, raw_data=True, cuda=True, batch_size: int = 1024, lr: float = 5e-4, epochs: int = 4,
        diffusion_steps: int = 3, repeats=3, with_benchmark=False, ctgan_epochs=30, metaseed=42,
        save_gen_data=True):
    # Settings
    experiment_name: str = f"exp_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    generated_data_path = Path('data/output_data') / experiment_name
    generated_data_path.mkdir(parents=True, exist_ok=True)
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    # Random seed for each repeat
    np.random.seed(metaseed)
    _seeds = np.random.randint(10000, size=repeats)

    # Git and MLFlow settings
    repo = git.Repo("./")
    githash = str(repo.head.object.hexsha)
    gitmessage = str(repo.head.object.message)
    gitbranch = str(repo.active_branch)
    git_remote_url = str(repo.remotes.origin.url)

    client = MlflowClient()
    client.set_experiment_tag(experiment_id, "mlflow.note.content", f"{gitbranch}:{githash[:6]}:{gitmessage}")
    dataset_path = Path('./') / dataset['path']
    for repeat in tqdm(range(1, repeats + 1), position=0, leave=True, desc="Repeats", colour="blue"):
        _seed = _seeds[repeats - 1]
        set_random_seed(_seed)

        X, _X, processor = load_and_prep_data(datadir='data/input_data')
        discrete_cols = [c for c, dtype in dataset["data_types"] if "categorical" in dtype]

        if with_benchmark:
            print("Benchmarking with CTGAN...")
            ctgan = CTGAN(epochs=ctgan_epochs, cuda=cuda)
            ctgan.fit(train_data=X, discrete_columns=discrete_cols)
            X_gen_benchmark = ctgan.sample(X.shape[0])
            print("Saving generated data...")
            pd.DataFrame(X_gen_benchmark).to_csv(Path(generated_data_path) / f"CTGAN_gen_{repeat}.csv", index=False)

        run_name = f"{experiment_name}_TableDiffusion_{repeat}"
        log_mlflow(run_name, experiment_id, githash, gitbranch, git_remote_url, gitmessage, repeat, repeats, _seed,
                   metaseed, _seeds, X, _X, discrete_cols, raw_data, cuda)

        print("Training model...")

        try:
            model = model(batch_size=batch_size, lr=lr, diffusion_steps=diffusion_steps, dims=(128, 128),
                          predict_noise=True)
            if raw_data:
                model = model.fit(X.copy(), discrete_columns=discrete_cols, n_epochs=epochs)
            else:
                model = model.fit(_X.copy(), discrete_columns=discrete_cols, n_epochs=epochs)

            mlflow.log_metrics({"elapsed_batches": model.elapsed_batches}, step=model.elapsed_batches)

            print("Generating data...")
            X_fake = pd.DataFrame(model.sample(_X.shape[0]))
            if not raw_data:
                # Reverse the transformation
                X_fake = pd.DataFrame(processor.inverse_transform(X_fake.values), columns=X.columns)

            evaluator = TableEvaluator(X, X_fake, cat_cols=discrete_cols, verbose=True)
            score = evaluator.column_correlations()

            # For the validation, we need to costumize the data types
            data_types = [
                ('device_label', 'categorical'),
                ('length', 'int'),
                ('IE 45', 'float'),
                ('IE 127', 'float'),
                ('IE 221', 'float'),
                ('IE 45*', 'categorical'),
                ('IE 127*', 'categorical'),
                ('IE 221*', 'categorical')]

            marginal_distance = compute_marginal_distances(X, X_fake, data_types)
            mlflow.log_metrics({f"{key.replace('*', '_')}md": value for key, value in marginal_distance.items()})

            mlflow.log_figure(evaluator.plot_mean_std(), "Log_mea_std.png")
            mlflow.log_figure(evaluator.plot_cumsums(), "cumsum.png")
            mlflow.log_figure(evaluator.plot_distributions(), "dist.png")
            mlflow.log_figure(evaluator.plot_pca(), "pca.png")
            mlflow.log_figure(evaluator.plot_correlation_difference(), "corr_difference.png")

            mlflow.log_metric("score", score)

            # Log samples and stats for fake data to MLflow
            pd.set_option("display.max_columns", 200)
            mlflow.log_text(
                str(X_fake.describe(include="all")), f"generated_data_info_{run_name}.txt",
            )

            if save_gen_data:
                print("Saving diffusion-generated data...")
                X_fake.to_csv(
                    Path(generated_data_path)
                    / f"genData_{run_name}_{repeat}.csv",
                    index=False,
                )
            return score

        except Exception as e:
            mlflow.set_tag("error", f"{e}\t{e.args}")
            raise e

        finally:
            mlflow.end_run()
