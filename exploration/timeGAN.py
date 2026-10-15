import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from hastyplot import qplot 
    import numpy as np
    import pandas as pd

    return mo, np, pd, qplot


@app.cell
def _(pd):
    #sim = pd.read_parquet('./stochastic_sim_v2_output.parquet')
    real = pd.read_parquet('./dataset.parquet')
    return (real,)


@app.cell
def _(real):
    real
    return


@app.cell
def _(real):
    _g = real.groupby('ramp_pattern_name')
    _g['frame'].max(), _g['uid'].count()
    return


@app.cell
def _(mo, qplot, real):
    _df = real.groupby(['uid', 'ramp_pattern_name'])['frame'].count()
    _df = _df.reset_index()
    mo.ui.altair_chart(
        qplot(data=_df, x='frame', facet_col='ramp_pattern_name')
    )
    return


@app.cell
def _(real):
    real.info()
    return


@app.cell(disabled=True)
def _():
    #class ERKWindowDataset(Dataset):
        # """
        # Per-cell sliding window dataset over ERK trajectory data.

        # Each sample is a fixed-length window of shape (window_size, n_features)
        # drawn from a single cell's (uid) time series, sorted by frame.

        # Args:
        #     df:             Source dataframe with at least `uid`, `frame`, and all
        #                     columns in feature_columns.
        #     feature_columns: Ordered list of column names to use as features.
        #     window_size:    Number of consecutive frames per sample.
        #     stride:         Step size between window starts (default 1).
        # """

        # def __init__(
        #     self,
        #     df: pd.DataFrame,
        #     feature_columns: list[str],
        #     window_size: int,
        #     stride: int = 1,
        # ):
        #     self.feature_columns = feature_columns
        #     self.window_size = window_size

        #     # Build index: list of (array_of_features, start_idx) per valid window
        #     self._windows: list[tuple[np.ndarray, int]] = []

        #     for uid, group in df.groupby("uid"):
        #         # Sort chronologically within each cell
        #         traj = (
        #             group.sort_values("frame")[feature_columns]
        #             .to_numpy(dtype=np.float32)
        #         )

        #         # Slide windows; cells shorter than window_size are silently skipped
        #         for start in range(0, len(traj) - window_size + 1, stride):
        #             self._windows.append((traj, start))

        # def __len__(self) -> int:
        #     return len(self._windows)

        # def __getitem__(self, idx: int) -> torch.Tensor:
        #     traj, start = self._windows[idx]
        #     window = traj[start : start + self.window_size]  # (window_size, n_features)
        #     return torch.from_numpy(window)
    return


@app.cell(hide_code=True)
def _():
    from erkdataset import ERKWindowDataset

    return (ERKWindowDataset,)


@app.cell(hide_code=True)
def _():
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset, DataLoader

    FEATURE_COLUMNS = """
    fluence_mJ_cm2
    ewma_fast
    cnr_mean_norm
    n_5
    median_cnr_0_9
    """.split()
    FEATURE_COLUMNS
    return (FEATURE_COLUMNS,)


@app.cell
def _(ERKWindowDataset, FEATURE_COLUMNS, real, torch):
    EPOCHS = 5
    SEQ_LEN = 25
    dataset = ERKWindowDataset(
        df=real,
        feature_columns=FEATURE_COLUMNS,
        window_size=SEQ_LEN,
        stride=13,
        stratify=True,
        response_col="cnr_mean_norm",
        stim_col="fluence_mJ_cm2",
        n_response_bins=3,
        n_stim_bins=3,
    )

    # Balanced batches: each window is drawn with weight 1/|stratum|, so in
    # expectation a batch sees equal numbers from each (response, stimulus)
    # bucket regardless of how lopsided the raw distribution is.
    balanced_sampler = dataset.make_balanced_sampler()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        sampler=balanced_sampler,
        num_workers=4,
    )
    # Each batch: (128, SEQ_LEN, n_features)
    return EPOCHS, SEQ_LEN, dataset, loader


@app.cell
def _(dataset, pd):
    # Per-stratum window counts. Labels read as r{response_bin}_s{stim_bin},
    # where response_bin is the cnr_mean_norm peak-to-peak quantile and
    # stim_bin is the fluence_mJ_cm2 mean quantile.
    _counts = dataset.stratum_counts()
    _stratum_count_df = pd.DataFrame(
        {
            "stratum": list(range(dataset.n_strata)),
            "stratum_label": [dataset.stratum_label(i) for i in range(dataset.n_strata)],
            "n_windows": _counts,
        }
    )
    _stratum_count_df
    return


@app.cell
def _(dataset):
    # Bin edges actually used (after pd.qcut drops duplicate quantiles for ties).
    {
        "response_edges (cnr_mean_norm ptp)": dataset.response_edges.tolist(),
        "stim_edges (fluence_mJ_cm2 mean)": dataset.stim_edges.tolist(),
    }
    return


@app.cell
def _(dataset):
    strata_samples = dataset.sample_per_stratum(n_per_stratum=10, seed=0)
    return (strata_samples,)


@app.cell
def _(mo, strata_samples):
    import altair as alt

    # cnr_mean_norm — the response axis. Each line is one sampled window;
    # facets are strata, ordered so response amplitude increases left→right
    # within each row and stimulus dose increases row-by-row.
    _resp_chart = (
        alt.Chart(strata_samples)
        .mark_line(opacity=0.55)
        .encode(
            x=alt.X("t:Q", title="frame within window"),
            y=alt.Y("cnr_mean_norm:Q", title="cnr_mean_norm"),
            color=alt.Color("sample_id:N", legend=None),
            detail="window_idx:N",
        )
        .properties(width=160, height=110)
        .facet(facet=alt.Facet("stratum_label:N", title="stratum"), columns=3)
        .properties(title="cnr_mean_norm per stratum (10 samples each)")
    )
    mo.ui.altair_chart(_resp_chart)
    return (alt,)


@app.cell
def _(alt, mo, strata_samples):
    # fluence_mJ_cm2 — the stimulus axis. Useful sanity check that the
    # stim-bin axis really separates dose levels.
    _stim_chart = (
        alt.Chart(strata_samples)
        .mark_line(opacity=0.55)
        .encode(
            x=alt.X("t:Q", title="frame within window"),
            y=alt.Y("fluence_mJ_cm2:Q", title="fluence_mJ_cm2"),
            color=alt.Color("sample_id:N", legend=None),
            detail="window_idx:N",
        )
        .properties(width=160, height=110)
        .facet(facet=alt.Facet("stratum_label:N", title="stratum"), columns=3)
        .properties(title="fluence_mJ_cm2 per stratum (10 samples each)")
    )
    mo.ui.altair_chart(_stim_chart)
    return


@app.cell
def _():

    import torch.nn as nn
    import torch.functional as F
    import torch
    from tqdm import tqdm

    return nn, torch


@app.cell
def _(torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    device
    return (device,)


@app.cell
def _(SEQ_LEN, device, nn, torch):
    class SubNetwork(nn.Module):
        def __init__(self, in_dim, out_dim, n_layers=2, final_activation=nn.Sigmoid):
            super().__init__()
            self.n_layers = n_layers
            self.rnn = nn.GRU(in_dim, out_dim, num_layers=n_layers, batch_first=True)
            self.fc = nn.Linear(out_dim, out_dim)
            self.nl = final_activation()

        def forward(self, input, h0=None):
            if h0 is not None:
                o, _ = self.rnn(input, h0)
            else:
                o, _ = self.rnn(input)
            return self.nl(self.fc(o))


    class TimeGAN(nn.Module):
        def __init__(self, data_dim, latent_dim, seq_len=SEQ_LEN):
            super().__init__()
            self.data_dim = data_dim
            self.latent_dim = latent_dim
            self.seq_len = seq_len
            self.device = device

            self.encoder = SubNetwork(data_dim, latent_dim)
            # Recovery output is unbounded — features here aren't guaranteed [0,1].
            self.recovery = SubNetwork(latent_dim, data_dim, final_activation=nn.Identity)
            self.generator = SubNetwork(data_dim, latent_dim)
            self.supervisor = SubNetwork(latent_dim, latent_dim)
            self.discriminator = SubNetwork(latent_dim, 1)
            self.to(device)

        def _sample_noise(self, batch_size, seq_len=None):
            seq_len = seq_len or self.seq_len
            return torch.randn(batch_size, seq_len, self.data_dim, device=self.device)

        def fit(
            self,
            train_dl: torch.utils.data.DataLoader,
            epochs_embed: int = 5,
            epochs_supervise: int = 5,
            epochs_joint: int = 5,
            bar=None,
        ):
            bce = nn.BCELoss()
            mse = nn.MSELoss()

            opt_er = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.recovery.parameters())
            )
            opt_s = torch.optim.Adam(self.supervisor.parameters())
            opt_g = torch.optim.Adam(
                list(self.generator.parameters()) + list(self.supervisor.parameters())
            )
            opt_e_joint = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.recovery.parameters())
            )
            opt_d = torch.optim.Adam(self.discriminator.parameters())

            losses = {"embed": [], "supervise": [], "joint": []}

            # ----- Stage 1: encoder + recovery on reconstruction -----
            for e in range(epochs_embed):
                for x in train_dl:
                    x = x.to(self.device)
                    h = self.encoder(x)
                    x_tilde = self.recovery(h)
                    l_rec = mse(x_tilde, x)
                    opt_er.zero_grad()
                    l_rec.backward()
                    opt_er.step()
                    losses["embed"].append({"rec": l_rec.item()})
                    if bar is not None:
                        bar.update(
                            increment=1,
                            title=f"[1/3 Embed] Ep {e+1}",
                            subtitle=f"rec={l_rec.item():.4f}",
                        )

            # ----- Stage 2: supervisor on next-step prediction in real latent space -----
            for e in range(epochs_supervise):
                for x in train_dl:
                    x = x.to(self.device)
                    with torch.no_grad():
                        h = self.encoder(x)
                    h_pred = self.supervisor(h[:, :-1, :])
                    l_sup = mse(h_pred, h[:, 1:, :])
                    opt_s.zero_grad()
                    l_sup.backward()
                    opt_s.step()
                    losses["supervise"].append({"sup": l_sup.item()})
                    if bar is not None:
                        bar.update(
                            increment=1,
                            title=f"[2/3 Sup] Ep {e+1}",
                            subtitle=f"sup={l_sup.item():.4f}",
                        )

            # ----- Stage 3: joint adversarial training (separate G, E, D steps) -----
            for e in range(epochs_joint):
                for x in train_dl:
                    x = x.to(self.device)
                    bs = x.shape[0]

                    # G step: fool discriminator + stay aligned with real latent dynamics
                    z = self._sample_noise(bs)
                    e_hat = self.generator(z)
                    h_hat = self.supervisor(e_hat)
                    y_fake = self.discriminator(h_hat)
                    l_g_adv = bce(y_fake, torch.ones_like(y_fake))

                    h = self.encoder(x)
                    h_sup_pred = self.supervisor(h[:, :-1, :])
                    l_g_sup = mse(h_sup_pred, h[:, 1:, :])

                    l_g = l_g_adv + 100.0 * torch.sqrt(l_g_sup + 1e-8)
                    opt_g.zero_grad()
                    l_g.backward()
                    opt_g.step()

                    # E step: keep recovery faithful; nudge encoder toward predictable dynamics
                    h = self.encoder(x)
                    x_tilde = self.recovery(h)
                    l_e_rec = mse(x_tilde, x)
                    h_sup_pred = self.supervisor(h[:, :-1, :])
                    l_e_sup = mse(h_sup_pred, h[:, 1:, :])
                    l_e = 10.0 * torch.sqrt(l_e_rec + 1e-8) + 0.1 * l_e_sup
                    opt_e_joint.zero_grad()
                    l_e.backward()
                    opt_e_joint.step()

                    # D step
                    with torch.no_grad():
                        h_real = self.encoder(x)
                        z_d = self._sample_noise(bs)
                        h_hat_d = self.supervisor(self.generator(z_d))
                    y_real = self.discriminator(h_real)
                    y_fake_d = self.discriminator(h_hat_d)
                    l_d = bce(y_real, torch.ones_like(y_real)) + bce(
                        y_fake_d, torch.zeros_like(y_fake_d)
                    )
                    opt_d.zero_grad()
                    l_d.backward()
                    opt_d.step()

                    losses["joint"].append(
                        {
                            "rec": l_e_rec.item(),
                            "sup": l_g_sup.item(),
                            "g_adv": l_g_adv.item(),
                            "d": l_d.item(),
                        }
                    )
                    if bar is not None:
                        bar.update(
                            increment=1,
                            title=f"[3/3 Joint] Ep {e+1}",
                            subtitle=f"rec={l_e_rec.item():.3f} sup={l_g_sup.item():.3f} g={l_g_adv.item():.3f} d={l_d.item():.3f}",
                        )

            self.losses = losses
            return losses

        def generate(self, batch_size, seq_len=None):
            seq_len = seq_len or self.seq_len
            self.eval()
            with torch.no_grad():
                z = self._sample_noise(batch_size, seq_len)
                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                x_hat = self.recovery(h_hat)
            return x_hat


    return (TimeGAN,)


@app.cell
def _(EPOCHS, FEATURE_COLUMNS, TimeGAN, loader, mo):
    _tg = TimeGAN(len(FEATURE_COLUMNS), 3)
    with mo.status.progress_bar(total=3 * EPOCHS * len(loader)) as _bar:
        final_losses = _tg.fit(
            loader,
            epochs_embed=EPOCHS,
            epochs_supervise=EPOCHS,
            epochs_joint=EPOCHS,
            bar=_bar,
        )
    return (final_losses,)


@app.cell
def _(final_losses, pd, qplot):
    _rows = []
    for _stage, _entries in final_losses.items():
        for _i, _ent in enumerate(_entries):
            for _k, _v in _ent.items():
                _rows.append(
                    {"stage": _stage, "iter": _i, "facet": f"{_stage}:{_k}", "value": _v}
                )
    _df_long = pd.DataFrame(_rows)
    qplot(_df_long, "iter", "value", facet_col="facet")
    return


if __name__ == "__main__":
    app.run()
