import warnings

import mlflow
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from .architectures import Generator
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from utilities.utils import gather_object_params

# Ignore warnings
warnings.filterwarnings("ignore")


# Function to compute the cosine noise schedule
def get_beta(t, T):
    return (1 - np.cos((np.pi * t) / T)) / 2 + 0.1


class MixedTypeGenerator(Generator):
    def __init__(
            self,
            embedding_dim,
            data_dim,
            gen_dims=(256, 256),
            predict_noise=True,
            categorical_start_idx=None,
            cat_counts=None,
    ):
        super().__init__(embedding_dim, data_dim, gen_dims)
        self.categorical_start_idx = categorical_start_idx
        self.cat_counts = cat_counts
        self.predict_noise = predict_noise

    def forward(self, x):
        data = self.seq(x)

        if self.predict_noise:
            return data  # Just predicting gaussian noise

        # Split into numerical and categorical outputs
        numerical_outputs = data[:, : self.categorical_start_idx]
        categorical_outputs = data[:, self.categorical_start_idx:]
        _idx = 0
        # Softmax over each category
        for k, v in self.cat_counts.items():
            categorical_outputs[:, _idx: _idx + v] = torch.softmax(
                categorical_outputs[:, _idx: _idx + v], dim=-1
            )
            _idx += v
        return torch.cat((numerical_outputs, categorical_outputs), dim=-1)


class TableDiffusion:
    def __init__(
            self,
            batch_size=1024,
            lr=0.005,
            b1=0.5,
            b2=0.999,
            dims=(128, 128),
            diffusion_steps=5,
            predict_noise=True,
            sample_img_interval=None,
            mlflow_logging=True,
            cuda=True,
    ):
        self.data_n = None
        self.data_dim = None
        self.encoded_columns = None
        self.total_categories = None
        self.disc_columns = None
        self._original_columns = None
        self._original_types = None
        self.category_counts = None
        self.encoders = None
        self.q_transformers = None
        from datetime import datetime
        self._now = datetime.now().strftime("%m%d%H%M%S")
        # Setting up GPU
        self.cuda = cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        # Hyperparameters
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.dims = dims
        self.diffusion_steps = diffusion_steps
        self.pred_noise = predict_noise
        self.sample_img_interval = sample_img_interval

        # Logging to MLflow
        self.mlflow_logging = mlflow_logging
        # if self.mlflow_logging:
        #     _param_dict = gather_object_params(self, prefix="init.")
        #     mlflow.log_params(_param_dict)

        self.elapsed_batches = 0
        self.elapsed_epochs = 0

    def fit(self, df, n_epochs=10, discrete_columns=None, verbose=False):

        if discrete_columns is None:
            discrete_columns = []
        global categorical_start_idx, loss
        self.data_dim = df.shape[1]
        self.data_n = df.shape[0]
        self.disc_columns = discrete_columns

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        self.q_transformers = {}
        self.encoders = {}
        self.category_counts = {}

        self._original_types = df.dtypes
        self._original_columns = df.columns
        df_encoded = df.select_dtypes(include="number").copy()
        df_encoded_cat = pd.DataFrame()
        for col in df.columns:
            if col in self.disc_columns:
                self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                transformed = self.encoders[col].fit_transform(df[col].values.reshape(-1, 1))
                transformed_df = pd.DataFrame(
                    transformed, columns=[f"{col}_{i}" for i in range(transformed.shape[1])]
                )
                df_encoded_cat = pd.concat([df_encoded_cat, transformed_df], axis=1)
                self.category_counts[col] = transformed_df.shape[1]
                categorical_start_idx = transformed_df.shape[1] + 1
            else:
                self.q_transformers[col] = QuantileTransformer()
                df_encoded[col] = self.q_transformers[col].fit_transform(
                    df[col].values.reshape(-1, 1)
                )
        df_encoded = pd.concat([df_encoded, df_encoded_cat], axis=1)

        self.total_categories = sum(self.category_counts.values())
        self.encoded_columns = df_encoded.columns
        self.data_dim = df_encoded.shape[1]
        self.data_n = df_encoded.shape[0]

        train_data = DataLoader(
            TensorDataset(torch.from_numpy(df_encoded.values.astype(np.float32))),
            batch_size=self.batch_size,
            drop_last=False,
        )

        self.model = MixedTypeGenerator(
            df_encoded.shape[1],
            self.data_dim,
            self.dims,
            self.pred_noise,
            categorical_start_idx,
            self.category_counts,
        ).to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=[0, 1, 2, 3])
        self.model.to(self.device)
        if verbose:
            print(self.model)

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2),
        )

        mse_loss = nn.MSELoss()
        kl_loss = nn.KLDivLoss(reduction="batchmean")

        self.model.train()

        for epoch in range(n_epochs):
            self.elapsed_epochs += 1
            for i, X in enumerate(train_data):
                X = X[0] # Because a DataLoader returns a tuple
                if i > 2 and loss.isnan():
                    print("Loss is NaN. Early stopping.")
                    return self

                if self.sample_img_interval is not None and i % self.sample_img_interval == 0:
                    fig, axs = plt.subplots(self.diffusion_steps, 5, figsize=(4 * self.diffusion_steps, 4 * 5))

                self.elapsed_batches += 1

                real_X = Variable(X.type(Tensor))
                agg_loss = torch.Tensor([0]).to(self.device)

                for t in range(self.diffusion_steps):
                    beta_t = get_beta(t, self.diffusion_steps)
                    noise = torch.randn_like(real_X).to(self.device) * np.sqrt(beta_t)
                    noised_data = real_X + noise

                    if self.sample_img_interval is not None and i % self.sample_img_interval == 0:
                        print(f"Epoch {epoch} (batch {i}, {t=}), {np.sqrt(beta_t)=}")

                    if self.pred_noise:
                        predicted_noise = self.model(noised_data)
                        numeric_loss = mse_loss(predicted_noise, noise)
                        categorical_loss = torch.tensor(0.0)
                        loss = numeric_loss

                    else:
                        denoised_data = self.model(noised_data)
                        numeric_loss = mse_loss(
                            denoised_data[:, :categorical_start_idx],
                            real_X[:, :categorical_start_idx],
                        )
                        _idx = categorical_start_idx
                        categorical_losses = []
                        for _col, _cat_len in self.category_counts.items():
                            categorical_losses.append(
                                kl_loss(
                                    torch.log(denoised_data[:, _idx: _idx + _cat_len]),
                                    real_X[:, _idx: _idx + _cat_len],
                                )
                            )
                            _idx += _cat_len

                        categorical_loss = (
                            sum(categorical_losses) / self.total_categories
                            if categorical_losses
                            else 0
                        )

                        loss = numeric_loss + categorical_loss

                    if self.sample_img_interval is not None and i % self.sample_img_interval == 0:
                        with torch.no_grad():
                            ax = axs[t]
                            ax[0].imshow(X.clone().detach().cpu().numpy())
                            ax[0].set_title("X")
                            ax[1].imshow(noise.clone().detach().cpu().numpy())
                            ax[1].set_title(f"noise_{t}")
                            ax[2].imshow(noised_data.clone().detach().cpu().numpy())
                            ax[2].set_title(f"noised_data_{t}")
                            if self.pred_noise:
                                ax[3].imshow(predicted_noise.clone().detach().cpu().numpy())
                                ax[3].set_title(f"predicted_noise_{t}")
                                denoised_data = noised_data - predicted_noise * np.sqrt(beta_t)
                            ax[4].imshow(denoised_data.clone().detach().cpu().numpy())
                            ax[4].set_title(f"denoised_data_{t}")

                    agg_loss += loss

                loss = agg_loss / self.diffusion_steps
                #print(f"Batches: {self.elapsed_batches}, {agg_loss=}")

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.sample_img_interval is not None and i % self.sample_img_interval == 0:
                    plt.savefig(
                        f"../results/diffusion_figs/{self._now}_forward_T{self.diffusion_steps}_B{self.elapsed_batches}.png")
                    sample = self.sample(n=X.shape[0], post_process=False)
                    plt.cla()
                    plt.clf()
                    plt.imshow(sample)
                    plt.savefig(
                        f"../results/diffusion_figs/{self._now}_sample_T{self.diffusion_steps}_B{self.elapsed_batches}.png")

                if i % 20 == 0:
                    if verbose:
                        print(
                            f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_data)}] numerical loss: {numeric_loss.item():.6f}, categorical loss: {categorical_loss.item():.6f}"
                        )
                    if self.mlflow_logging:
                        mlflow.log_metrics(
                            {
                                "elapsed_batches": self.elapsed_batches,
                                "elapsed_epochs": self.elapsed_epochs,
                                "train_loss.numerical": numeric_loss.item(),
                                "train_loss.categorical": categorical_loss.item(),
                                "train_loss.total": loss.item(),
                            },
                            step=self.elapsed_epochs,
                        )

                        # _norm_dict = calc_norm_dict(self.model)
                        # mlflow.log_metrics(_norm_dict, step=self._elapsed_batches)

        return self

    def sample(self, n=None, post_process=True):
        self.model.eval()
        n = self.batch_size if n is None else n
        samples = torch.randn((n, self.data_dim)).to(self.device)
        #fig, axs = plt.subplots(self.diffusion_steps, 4, figsize=(4 * self.diffusion_steps, 4 * 4))

        with torch.no_grad():
            for t in range(self.diffusion_steps - 1, -1, -1):
                beta_t = get_beta(t, self.diffusion_steps)
                noise_scale = np.sqrt(beta_t)
                #print(f"Sampling {t=}, {np.sqrt(beta_t)=}")
                #ax = axs[self.diffusion_steps - t - 1]
                #ax[2].imshow(samples.clone().detach().cpu().numpy())
                #ax[2].set_title(f"samples_{t}")

                if self.pred_noise:
                    pred_noise = self.model(samples)
                    predicted_noise = pred_noise * noise_scale
                    #ax[0].imshow(pred_noise.clone().detach().cpu().numpy())
                    #ax[0].set_title(f"pred_noise_{t}")
                    #ax[1].imshow(predicted_noise.clone().detach().cpu().numpy())
                    #ax[1].set_title(f"predicted_noise_{t}")

                    samples = samples - predicted_noise
                else:
                    samples = self.model(samples)
                #ax[3].imshow(samples.clone().detach().cpu().numpy())
                #ax[3].set_title(f"samples_{t - 1}")

        if self.sample_img_interval is not None:
            plt.savefig(
                f"../results/diffusion_figs/{self._now}_reverse_T{self.diffusion_steps}_B{self.elapsed_batches}.png")

        synthetic_data = samples.detach().cpu().numpy()
        self.model.train()

        if not post_process:
            return synthetic_data

        df_synthetic = pd.DataFrame(synthetic_data, columns=self.encoded_columns)
        for col in self.encoders:
            transformed_cols = [c for c in df_synthetic.columns if c.startswith(f"{col}_")]
            if transformed_cols:
                encoded_data = df_synthetic[transformed_cols].values
                df_synthetic[col] = self.encoders[col].inverse_transform(encoded_data).ravel()
                df_synthetic = df_synthetic.drop(columns=transformed_cols)

        for col in self.q_transformers:
            df_synthetic[col] = self.q_transformers[col].inverse_transform(
                df_synthetic[col].values.reshape(-1, 1)
            )

        df_synthetic = df_synthetic.astype(self._original_types)
        df_synthetic = df_synthetic[self._original_columns]

        return df_synthetic
