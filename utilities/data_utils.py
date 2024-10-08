import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

with open('utilities/datasets.json', 'r') as file:
    dataset = json.load(file)


def load_and_prep_data(datadir, verbose=False):
    """Loads the dataset. Returns X, X_tx, processor"""
    datadir = Path(datadir)
    if not os.path.exists(datadir):
        raise ValueError(f"'{datadir}' not found.")

    # Load the dataset
    path = Path(datadir) / dataset["path"]
    X = pd.read_csv(path, sep=dataset["sep"]).drop(dataset["drop_cols"], axis=1)
    if verbose:
        print(f"Loaded dataset with shape {X.shape} from {path}")

    # Transform dataset
    if verbose:
        print("Transforming dataset...")
    processor = DataProcessor(dataset["data_types"]).fit(X)
    X_tx = processor.transform(X.values)
    if verbose:
        print(f"Data is now {X_tx.shape}")

    # Transform the pandas columns into the correct data types
    for col, dtype in dataset["data_types"]:
        if dtype == "categorical":
            X[col] = X[col].astype("object")
        elif dtype == "numerical":
            X[col] = X[col].astype("float")
    if verbose:
        print(X.value_counts())

    return X, X_tx, processor


def calc_norm_dict(model, ord=1, errors="ignore", no_biases=True):
    """Calculate the norm of order `ord` for weights and gradients in
    each layer of model, returned as a dictionary.
    """
    _norm_dict = {}
    _mod_name = model.__class__.__name__
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if no_biases and "bias" in name:
            continue
        try:
            _norm_dict[f"{_mod_name}.{name}.weight_norm"] = parameter.data.norm(ord).item()
            _norm_dict[f"{_mod_name}.{name}.grad_norm"] = parameter.grad.norm(ord).item()
        except Exception as e:
            if errors != "ignore":
                raise e

    return _norm_dict


class DataProcessor:
    """
    Extended from:
    https://github.com/DPautoGAN/DPautoGAN/blob/master/uci/uci.ipynb
    """

    def __init__(self, datatypes):
        self.datatypes = datatypes

    def fit(self, df: pd.DataFrame):
        matrix = df.values

        preprocessors, cutoffs = [], []
        for i, (_, datatype) in enumerate(self.datatypes):
            preprocessed_col = matrix[:, i].reshape(-1, 1)

            if "categorical" in datatype:
                preprocessor = LabelBinarizer()
            else:
                preprocessor = MinMaxScaler()
            preprocessed_col = preprocessor.fit_transform(preprocessed_col)
            cutoffs.append(preprocessed_col.shape[1])
            preprocessors.append(preprocessor)

        self.cutoffs = cutoffs
        self.preprocessors = preprocessors
        self.col_names = [c for (c, _) in self.datatypes]

        return self

    def transform(self, data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            data = data.values

        preprocessed_cols = []

        for i, _ in enumerate(self.datatypes):
            preprocessed_col = data[:, i].reshape(-1, 1)
            preprocessed_col = self.preprocessors[i].transform(preprocessed_col)
            preprocessed_cols.append(preprocessed_col)

        return np.concatenate(preprocessed_cols, axis=1)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, data) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            data = data.values

        postprocessed_cols = []

        j = 0
        for i, (_, datatype) in enumerate(self.datatypes):
            postprocessed_col = self.preprocessors[i].inverse_transform(
                data[:, j: j + self.cutoffs[i]]
            )

            if "categorical" in datatype:
                postprocessed_col = postprocessed_col.reshape(-1, 1)
            else:
                if "positive" in datatype:
                    postprocessed_col = postprocessed_col.clip(min=0)

                if "int" in datatype:
                    postprocessed_col = postprocessed_col.round()

            postprocessed_cols.append(postprocessed_col)

            j += self.cutoffs[i]

        return pd.DataFrame(
            np.concatenate(postprocessed_cols, axis=1), columns=self.col_names
        ).apply(pd.to_numeric, errors="ignore")
