import os.path
import pathlib

import click
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, friedmanchisquare

from metraq_dip.experiments import get_experiment_name


def load_experiment_data(experiment_file: pathlib.Path):
    exp_data = np.load(experiment_file, allow_pickle=True)
    minmax_map = None
    if "minmax_map" in exp_data:
        minmax_map = exp_data["minmax_map"].item()

    # old experiments can use 'x_data' and 'y_data' instead of 'train_data' and 'test_data'
    train_data = exp_data['x_data' if 'x_data' in exp_data else 'train_data'].astype(float)
    test_data = exp_data['y_data' if 'y_data' in exp_data else 'test_data'].astype(float)
    train_mask = exp_data['train_mask'].astype(bool)
    val_mask = exp_data['val_mask'].astype(bool)
    test_mask = exp_data['test_mask'].astype(bool)

    mask = (train_mask | val_mask)
    data = np.max(train_data, axis=0)[mask.any(axis=0)==1]
    stats = {
        "obs_mean": data.mean(),
        "obs_median": np.median(data),
        "obs_max": data.max(),
        "obs_std": data.std(),
        "obs_p90_p10": np.percentile(data, 90) - np.percentile(data, 10),
    }

    experiment_data = {
        'train_data': train_data,
        'test_data': test_data,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask':test_mask,

        'train_output': exp_data['train_output'].astype(float),
        'val_min_idx': exp_data['val_min_idx'].astype(int),
        'train_k_output': exp_data['train_k_output'].astype(float),
        'train_k_loss': exp_data['train_k_loss'].astype(float),
        'val_k_loss': exp_data['val_k_loss'].astype(float),
        'minmax_map': minmax_map,
        'data_stats': stats
    }

    return experiment_data


def verify_session(session_folder: str, verbose: bool = True):
    result = {}

    path = pathlib.Path(session_folder)
    if not path.exists():
        if verbose:
            click.secho("Session folder {session_folder} does not exist.", fg="red", err=True)
    else:
        if verbose:
            click.secho(f"Session folder {session_folder} exists.", fg="green")

    # load configuration
    config_file = path.joinpath("config.yaml")
    if not config_file.exists():
        if verbose:
            click.secho("Session folder {session_folder} does not contain a config.json file.", fg="red",
                err=True)
    else:
        if verbose:
            click.secho(f"Session folder {session_folder} contains a config.json file.", fg="green")

        result['config_file'] = config_file

    # load original data
    data_file = path.joinpath("data.npz")
    if not data_file.exists():
        if verbose:
            click.secho("Session folder {session_folder} does not contain a data.npz file.", fg="red", err=True)
    else:
        if verbose:
            click.secho(f"Session folder {session_folder} contains a data.npz file.", fg="green")
        result['data_file'] = data_file

    experiment_data_files = path.glob("exp_*.npz")
    results_file = path.joinpath("results.csv")

    if experiment_data_files and not results_file.exists():
        if verbose:
            click.secho(f"Session folder {session_folder} contains experiment data files but no results file.",
                        fg="red", err=True)
    elif experiment_data_files:
        result['experiment_data_files'] = list(experiment_data_files)
        session_results = pd.read_csv(results_file, parse_dates=["time_window"])

        for idx, row in session_results.iterrows():
            experiment_file_name = f"{get_experiment_name(row['sensor_group'], row['time_window'])}.npz"
            if not pathlib.Path(session_folder).joinpath(experiment_file_name).exists():
                if verbose:
                    click.secho("Warning: Experiment file {experiment_file_name} for result {idx} does not exist.",
                            fg="red", err=True)

        if results_file.exists():
            if verbose:
                click.secho(f"Session folder {session_folder} contains experiment data files and a results file.",
                        fg="green")
            result['results_file'] = results_file

    return result


def load_training_session(session_folder: str, load_experiments: bool = False):
    session_files = verify_session(session_folder, verbose=False)

    # load configuration
    if not "config_file" in session_files:
        raise ValueError(f"Session folder {session_folder} does not contain a config.json file.")

    config_file =session_files["config_file"]
    config = yaml.safe_load(config_file.read_text())

    # load original data
    if not "data_file" in session_files:
        raise ValueError(f"Session folder {session_folder} does not contain a data.npz file.")

    data_file = session_files["data_file"]
    data = np.load(data_file, allow_pickle=True)
    test_sensors = data["test_sensors"]
    time_windows = data["time_windows"]

    # load experiments data
    if not "experiment_data_files" in session_files:
        raise ValueError(f"Session folder {session_folder} does not contain any experiment data files.")

    experiment_data_files = session_files["experiment_data_files"]

    return_data = {
        'configuration': config,
        'data': {
            'test_sensors': test_sensors,
            'time_windows': time_windows,
        },
    }

    if  'experiment_data_files' in session_files:
        return_data['experiment_files'] = experiment_data_files

        if load_experiments:
            exp = {}
            for file in experiment_data_files:
                experiment_data = load_experiment_data(file)
                exp[file.stem] = experiment_data

            return_data['experiments'] = exp

    # load results
    if 'results_file' in session_files:
        experiments_results = pd.read_csv(session_files['results_file'], parse_dates=["time_window"])
        return_data['results'] = experiments_results

    return return_data


def get_session_results(session_data: dict):
    results = session_data['results']
    dip_loss = results['DIP_L1Loss'].values
    kriging_loss = results['KRG_L1Loss'].values
    idw_loss = results['IDW_L1Loss'].values

    # Calculate Friedman's chi-squared test'
    stat, p_value = friedmanchisquare(dip_loss, kriging_loss, idw_loss)

    # Calculate mean ranks
    data = np.vstack([dip_loss, kriging_loss, idw_loss]).T
    ranks = np.apply_along_axis(rankdata, 1, data)
    mean_ranks = np.mean(ranks, axis=0)

    return {
        'friedman': (stat, p_value),
        'mean_ranks': mean_ranks,
    }

def get_experiment_result(experiment: dict):
    t_data = experiment['train_data'][0]
    train_mask = experiment['train_mask']
    val_mask = experiment['val_mask']
    test_mask = experiment['test_mask']
    k_output = experiment['train_k_output']
    count = test_mask.sum()

    train_data = t_data * train_mask.astype(int)
    val_data = t_data * val_mask.astype(int)
    test_data = experiment['test_data'][0]

    best_k = 10
    k, c, t, h, w = k_output.shape
    for i in range(k):
        y_hat = k_output[i, 0] * test_mask
        diff = np.abs(y_hat - test_data)
        l1_losses = np.sum(diff, axis=(1, 2)) / count
        mse_losses = np.sum(diff**2, axis=(1, 2)) / count
        best_l1_idx = np.argpartition(l1_losses, best_k)[:best_k]
        best_l1_losses = l1_losses[best_l1_idx]
        pass


if __name__ == "__main__":
    experiment_folder = "output/experiments/norm/experiment_baseline"
    session = load_training_session(experiment_folder, load_experiments=True)
    results = session['results']

    for experiment_key, experiment in session['experiments'].items():
        _, sensor_group, time_window = experiment_key.split("_")
        mask = (
                (results['sensor_group'] == sensor_group) &
                (results['time_window'] == time_window)
        )

        results.loc[mask, 'data_mean'] = experiment['data_stats']['obs_mean']
        results.loc[mask, 'data_median'] = experiment['data_stats']['obs_median']
        results.loc[mask, 'data_max'] = experiment['data_stats']['obs_max']
        results.loc[mask, 'data_std'] = experiment['data_stats']['obs_std']
        results.loc[mask, 'data_p90_p10'] = experiment['data_stats']['obs_p90_p10']

    results.to_csv(os.path.join(experiment_folder, "results_with_stats.csv"), index=False)



