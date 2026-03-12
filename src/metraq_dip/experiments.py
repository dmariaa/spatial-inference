import hashlib
import os
from datetime import datetime
from typing import Optional

import click
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from metraq_dip.tools.random_tools import get_spread_test_groups, get_random_time_windows, sensor_group_hash
from metraq_dip.tools.tools import get_interpolation_loss
from metraq_dip.trainer.trainer_dip import DipTrainer


def get_experiment_name(sensor_group_key: str, time_window: datetime):
    return f"exp_{sensor_group_key}_{time_window.strftime('%Y%m%dT%H%M%S')}"

def run_experiments(*, config_base: Optional[dict] = None, output_folder: str = 'output',
                    experiment_name: str = None):
    experiment_name = experiment_name if experiment_name else f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_output_folder = os.path.join(output_folder, experiment_name)

    # Reload existing or save new configuration
    if os.path.exists(experiment_output_folder) and os.path.exists(os.path.join(experiment_output_folder, "config.yaml")):
        print(f"Experiment {experiment_name} already exists, loading configuration")
        with open(os.path.join(experiment_output_folder, "config.yaml"), "r") as f:
            saved_config = yaml.safe_load(f)

            if config_base:
                saved_config.update(config_base)

            config_base = saved_config
    else:
        print(f"Experiment {experiment_name} not found, generating a new one")
        os.makedirs(experiment_output_folder, exist_ok=True)

        with open(os.path.join(experiment_output_folder, "config.yaml"), "w") as f:
            yaml.dump(config_base, f)

    data_file = os.path.join(experiment_output_folder, "data.npz")
    if os.path.exists(data_file):
        click.echo(f"Experiment {experiment_name} already has data, skipping data generation")
        data = np.load(os.path.join(experiment_output_folder, "data.npz"), allow_pickle=True)
        test_sensors = data["test_sensors"]
        time_windows = data["time_windows"]
    else:
        click.echo(f"Generating data for experiment {experiment_name}")
        pollutants = config_base["pollutants"]
        # TODO: hardcoded get all this data from configuration
        test_sensors, _ = get_spread_test_groups(n_groups=10, group_size=4, max_uses_per_sensor=2, magnitudes=pollutants)
        time_windows = get_random_time_windows(year=2024, windows_per_month=20, start_hours=(8, 17))
        np.savez(os.path.join(experiment_output_folder, "data.npz"),
                 test_sensors=test_sensors,
                 time_windows=time_windows)

    if os.path.exists(os.path.join(experiment_output_folder, "results.csv")):
        df = pd.read_csv(os.path.join(experiment_output_folder, "results.csv"),
                         dtype={"sensor_group": "string"},
                         parse_dates=["time_window"])
    else:
        df = pd.DataFrame()
        time_window_series = pd.to_datetime(time_windows)
        sensor_group_keys = [sensor_group_hash(group) for group in test_sensors]
        rows = [(time_window, sensor_group) for sensor_group in sensor_group_keys for time_window in time_window_series]
        df = pd.DataFrame(rows, columns=["time_window", "sensor_group"])
        df["sensor_group"] = df["sensor_group"].astype("string")
        df["processed"] = False
        df['DIP_L1Loss'] = 0.0
        df['DIP_MSELoss'] = 0.0
        df['KRG_L1Loss'] = 0.0
        df['KRG_MSELoss'] = 0.0
        df['IDW_L1Loss'] = 0.0
        df['IDW_MSELoss'] = 0.0
        df.to_csv(os.path.join(experiment_output_folder, "results.csv"), index=False)

    # Run the experiments
    with tqdm(total=len(test_sensors)*len(time_windows), position=0) as pbar:
        for test_sensor_group in test_sensors:
            sensor_group_key = sensor_group_hash(test_sensor_group)

            for time_window in time_windows:
                row_mask = (df["time_window"] == pd.to_datetime(time_window)) & (df["sensor_group"] == sensor_group_key)
                if not df.loc[row_mask].empty and df.loc[row_mask, "processed"].any():
                    pbar.update(1)
                    continue

                pbar.set_description(desc=f"Experiment {sensor_group_key} - {time_window.isoformat()}")
                # setup experiment configuration
                config = config_base.copy()
                config["date"] = time_window.isoformat()
                config["validation_sensors"] = 4
                config["test_sensors"] = test_sensor_group

                # launch the trainer and gather results
                trainer = DipTrainer(configuration=config)
                trainer()
                result, output, min_idx_s = trainer.get_best_result()

                # gather interpolation results
                x_data = (trainer.dip_logger['train_data'] + trainer.dip_logger['val_data'])[0, :, -1:, ...].detach().cpu()
                y_data = trainer.dip_logger['test_data'][0, :, -1:, ...].detach().cpu()

                train_mask =  (trainer.dip_logger['train_mask'] + trainer.dip_logger['val_mask'])[0, :, -1:, ...].detach().cpu()
                test_mask = trainer.dip_logger['test_mask'][0, :, -1:, ...].detach().cpu()

                interpolation_results = get_interpolation_loss(
                    x_data,
                    train_mask,
                    y_data,
                    test_mask, trainer.pollutants)

                # save experiment results
                mask = (df["time_window"] == pd.to_datetime(time_window)) & (df["sensor_group"] == sensor_group_key)
                df.loc[mask, [
                    "sensor_group",
                    "processed",
                    "DIP_L1Loss",
                    "DIP_MSELoss",
                    "KRG_L1Loss",
                    "KRG_MSELoss",
                    "IDW_L1Loss",
                    "IDW_MSELoss"
                ]] = [
                    sensor_group_key,
                    True,
                    result[0]["loss"],
                    result[1]["loss"],
                    interpolation_results[0]["loss"],
                    interpolation_results[1]["loss"],
                    interpolation_results[2]["loss"],
                    interpolation_results[3]["loss"],
                ]

                df.to_csv(os.path.join(experiment_output_folder, "results.csv"), index=False)

                # save experiment data
                experiment_data = {
                    'train_data': trainer.dip_logger['train_data'].detach().cpu().numpy(),
                    'val_data': trainer.dip_logger['val_data'].detach().cpu().numpy(),
                    'test_data': trainer.dip_logger['test_data'].detach().cpu().numpy(),
                    'train_mask': trainer.dip_logger['train_mask'].detach().cpu().numpy(),
                    'val_mask': trainer.dip_logger['val_mask'].detach().cpu().numpy(),
                    'test_mask': trainer.dip_logger['test_mask'].detach().cpu().numpy(),

                    'train_output': output,
                    'val_min_idx': min_idx_s,

                    'train_k_output': trainer.dip_logger['train_output'].detach().cpu().numpy(),
                    'train_k_loss': trainer.dip_logger['train_loss'].detach().cpu().numpy(),
                    'val_k_loss': trainer.dip_logger['val_loss'].detach().cpu().numpy(),
                    'test_k_loss': trainer.dip_logger['test_loss'].detach().cpu().numpy(),
                }
                experiment_file_name = f"{get_experiment_name(sensor_group_key, time_window)}.npz"
                np.savez_compressed(os.path.join(experiment_output_folder, experiment_file_name), **experiment_data)

                pbar.update(1)


if __name__=="__main__":
    run_experiments(config_base=None, output_folder="output/experiments", experiment_name="experiment_test_6")
