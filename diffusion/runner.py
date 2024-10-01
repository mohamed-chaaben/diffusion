from datetime import datetime
from pathlib import Path
import mlflow
import pandas as pd
import torch
import numpy as np
from ctgan.synthesizers import CTGAN
from table_evaluator import TableEvaluator
from tqdm import tqdm

from utilities.data_utils import load_and_prep_data, dataset
from utilities.metrics import compute_marginal_distances
from utilities.utils import set_random_seed

from .table_diffusion import TableDiffusion


def log_mlflow(run_name, exp_id, repeat, repeats, _seed, metaseed, _seeds, X_shape, _X_shape, discrete_cols, raw_data,
               cuda):
    # Set MLflow tags for metadata tracking
    mlflow.set_tags({
        "run_name": run_name,  # Log the run name for easy tracking
        "repeat": f"{repeat}/{repeats}",
        "random_seed.run": _seed,
        "random_seed.meta": metaseed,
        "random_seed.seeds": str(_seeds),  # Stringify the list of seeds
    })

    # Log MLflow parameters
    try:
        gpu_properties = str(torch.cuda.get_device_properties(0)) if cuda else "cpu"
    except RuntimeError as e:
        gpu_properties = "N/A"  # Handle the case where GPU properties cannot be fetched

    mlflow.log_params({
        "gpu_properties": gpu_properties,
        "dataset.shape_raw": X_shape,
        "dataset.shape_transformed": _X_shape,
        "dataset.raw_data_used": raw_data,
        "dataset.discrete_cols": list(discrete_cols)
    })


def run(raw_data=True, cuda=True, batch_size: int = 1024, lr: float = 5e-4, epochs: int = 4,
        diffusion_steps: int = 3, repeats=3, with_benchmark=False, ctgan_epochs=30, metaseed=42, save_gen_data=True,
        experiment_id=None):
    # Settings
    assert experiment_id is not None, "Experiment ID must be provided."

    # Random seed for each repeat
    np.random.seed(metaseed)
    _seeds = np.random.randint(10000, size=repeats)
    scores = []
    for repeat in tqdm(range(1, repeats + 1), desc="Repeats", colour="blue"):
        _seed = _seeds[repeats - 1]
        set_random_seed(_seed)

        X, _X, processor = load_and_prep_data(datadir='data/input_data')
        discrete_cols = [c for c, dtype in dataset["data_types"] if "categorical" in dtype]

        # Define run name for the repeat
        run_name = f"{experiment_id}_Trial_{repeat}"
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id, nested=True):
            generated_data_path = Path('data/output_data') / experiment_id
            generated_data_path.mkdir(parents=True, exist_ok=True)

            if with_benchmark:
                print("Benchmarking with CTGAN...")
                ctgan = CTGAN(epochs=ctgan_epochs, cuda=cuda)
                ctgan.fit(train_data=X, discrete_columns=discrete_cols)
                X_gen_benchmark = ctgan.sample(X.shape[0])
                print("Saving generated data...")
                pd.DataFrame(X_gen_benchmark).to_csv(Path(generated_data_path) / f"CTGAN_gen_{run_name}.csv",
                                                     index=False)

            log_mlflow(run_name=run_name, exp_id=experiment_id, repeat=repeat, repeats=repeats, _seed=_seed,
                       metaseed=metaseed, _seeds=_seeds, X_shape=X.shape, _X_shape=_X.shape,
                       discrete_cols=discrete_cols,
                       raw_data=raw_data, cuda=cuda)

            print("Training model...")

            try:
                model = TableDiffusion(batch_size=batch_size, lr=lr, diffusion_steps=diffusion_steps, dims=(128, 128),
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
                scores.append(score)

            except Exception as e:
                mlflow.set_tag("error", f"{e}\t{e.args}")
                raise e

    return sum(scores) / len(scores) if scores else None
