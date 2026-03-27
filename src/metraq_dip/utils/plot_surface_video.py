from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from metraq_dip.tools.interpolator import IdwInterpolator, KrigingInterpolator
from metraq_dip.tools.tools import calculate_interpolations
from metraq_dip.trainer.tools import load_training_session


def plot_video(*, data: dict, limits: Tuple[float, float] = None, title: Optional[str] = None):
    fig = make_subplots( rows=1, cols=1, specs=[[{'type': 'surface'}]])
    plotly_3d_surface_video(data, fig, 1, 1, limits=limits, title=title)
    return fig


def plotly_3d_surface_video(data: dict, fig: go.Figure, row: int, col: int, limits: Tuple[float, float],
                            title: Optional[str] = None):
    # torch → numpy if needed
    y_hat = data['y']
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    test_loss = data.get('test_loss', None)

    min_train_loss = np.argmin(train_loss)
    min_val_loss = np.argmin(val_loss)
    min_test_loss = np.argmin(test_loss) if test_loss is not None else None

    T, H, W = y_hat.shape

    if limits is None:
        z_min = float(np.nanmin(y_hat))
        z_max = float(np.nanmax(y_hat))
    else:
        z_min, z_max = limits

    fig.add_trace(
        go.Surface(
            z=y_hat[0],
            colorscale="Viridis",
            cmin=z_min,
            cmax=z_max,
            opacity=0.85,
            colorbar=dict(
                x=1.0,  # horizontal position (default ~1.02)
                y=0.0,  # vertical center (0–1)
                yanchor="bottom",
                len=0.75,  # height relative to plot
                thickness=20,  # width in pixels
                title="concentration"
            )
        ),
        row=row,
        col=col
    )

    loss_annotation = dict(
                text="Test",  # will be filled by frames
                x=0.5,
                y=0.985,
                yanchor="bottom",
                xanchor="center",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, family="monospace"),
                align="center",
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1,
                borderpad=4,
            )

    fig.frames = [
             go.Frame(
                 data=[go.Surface(z=y_hat[t])],
                 name=str(t),
                 traces=[0],
                 layout=go.Layout(
                     annotations=[
                         dict(
                             loss_annotation,
                             text=(
                                     f"<b>Frame {t}</b>: "
                                     f"Train loss: {train_loss[t, 0]:.4f} - "
                                     f"Val loss: {val_loss[t, 0]:.4f}"
                                     + (
                                         f" - Test loss: {test_loss[t, 0]:.4f}"
                                         if test_loss is not None
                                         else ""
                                     )
                             ),
                         )
                     ]
                 )
             )
             for t in range(T)
         ]

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            y=0.985,
            xref="paper",
            xanchor="center",
            yanchor="top",
            font=dict(size=12),
        ),
        margin=dict(
            l=20,
            r=20,
            t=80,
            b=40
        ),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="concentration",
            zaxis=dict(range=[z_min, z_max]),
            aspectmode = "manual",
            aspectratio = dict(x=1, y=H / W, z=0.6),  # tune z to taste
        ),
        annotations=[loss_annotation],
        sliders=[{
            "active": 0,
            "x": 0.05, "y": 0.02,
            "len": 0.9,
            "currentvalue": {"prefix": "frame: "},
            "steps": [
                {"method": "animate",
                 "label": str(t),
                 "args": [[str(t)], {"mode": "immediate",
                                     "frame": {"duration": 0, "redraw": True},
                                     "transition": {"duration": 0}}]}
                for t in range(T)
            ]
        }],
        legend=dict(
            x=1.02,  # move left/right (0–1)
            y=0.9,  # move up/down (0–1)
            xanchor="left",  # anchor point of legend box
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    )

    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(
                x=-0.029866526312817884,
                y=-0.03655669711789127,
                z=-0.20813091351640828
            ),
            eye=dict(
                x=0.9544245505546257,
                y=-1.06399525960207,
                z=0.4811393131790605
            ),
            projection=dict(type="perspective")
        ),
        uirevision="camera"
    )

    return fig


def _denormalize_masked(data: np.ndarray, mask: np.ndarray, *, mean: float, std: float) -> np.ndarray:
    restored = np.array(data, copy=True, dtype=np.float32)
    valid_mask = np.asarray(mask, dtype=bool)
    restored[valid_mask] = restored[valid_mask] * (std + 1e-6) + mean
    restored[~valid_mask] = 0.0
    return restored


def _denormalize_full(data: np.ndarray, *, mean: float, std: float) -> np.ndarray:
    restored = np.array(data, copy=True, dtype=np.float32)
    return restored * (std + 1e-6) + mean


def _denormalize_losses(losses: np.ndarray, *, std: float) -> np.ndarray:
    restored = np.array(losses, copy=True, dtype=np.float32)
    scale = std + 1e-6
    restored[..., 0] = restored[..., 0] * scale
    if restored.shape[-1] > 1:
        restored[..., 1] = restored[..., 1] * (scale ** 2)
    return restored


def _compute_masked_losses(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid_mask = np.asarray(mask, dtype=bool)
    count = int(valid_mask.sum())
    if count == 0:
        return np.zeros(2, dtype=np.float32)

    diff = (np.asarray(y_pred, dtype=np.float32) - np.asarray(y_true, dtype=np.float32))[valid_mask]
    return np.array(
        [
            np.abs(diff).sum() / count,
            np.square(diff).sum() / count,
        ],
        dtype=np.float32,
    )


def _get_interpolator(method: str):
    if method == "kriging":
        return KrigingInterpolator
    return IdwInterpolator


def _build_dip_plot_data(
    *,
    train_output: np.ndarray,
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict:
    y_hat = np.asarray(train_output, dtype=np.float32)[None, ...]

    train_loss = _compute_masked_losses(train_data[0, 0, -1], y_hat[0], train_mask[0, 0, -1])[None, :]
    val_loss = _compute_masked_losses(val_data[0, 0, -1], y_hat[0], val_mask[0, 0, -1])[None, :]
    test_loss = _compute_masked_losses(test_data[0, 0, -1], y_hat[0], test_mask[0, 0, -1])[None, :]

    return {
        "y": y_hat,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
    }


def _build_baseline_plot_data(
    *,
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    method: str,
) -> dict:
    observed_data = train_data + val_data
    observed_mask = train_mask | val_mask
    interpolator = _get_interpolator(method)
    current_observed_data = observed_data[0, :, -1:, ...]
    current_observed_mask = observed_mask[0, :, -1:, ...]

    y_hat = calculate_interpolations(
        torch.as_tensor(current_observed_data, dtype=torch.float32),
        torch.as_tensor(current_observed_mask, dtype=torch.bool),
        interpolator,
    )[0, 0:1]

    train_loss = _compute_masked_losses(train_data[0, 0, -1], y_hat[0], train_mask[0, 0, -1])[None, :]
    val_loss = _compute_masked_losses(val_data[0, 0, -1], y_hat[0], val_mask[0, 0, -1])[None, :]
    test_loss = _compute_masked_losses(test_data[0, 0, -1], y_hat[0], test_mask[0, 0, -1])[None, :]

    return {
        "y": y_hat,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
    }


def add_gt_points(fig, gt_surface, mask, marker_size=5, name="ground truth", color:str="red"):
    gt_surface = np.asarray(gt_surface)
    mask = np.asarray(mask).astype(bool)

    ys, xs = np.where(mask)          # row (y), col (x)
    zs = gt_surface[ys, xs].astype(float)

    fig.add_trace(
        go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=marker_size, color=color),
            name=name
        )
    )
    return fig


def _build_plot_title(session_folder_name: str, experiment_file_name: str) -> str:
    stem = experiment_file_name.rsplit(".", 1)[0]
    if not stem.startswith("exp_"):
        return f"Experiment: {session_folder_name} | File: {experiment_file_name}"

    payload = stem[4:]
    sensor_group, sep, time_token = payload.rpartition("_")
    if sep == "":
        return f"Experiment: {session_folder_name} | File: {experiment_file_name}"

    try:
        formatted_time = datetime.strptime(time_token, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        formatted_time = time_token

    sensors_text = sensor_group.replace("-", ", ")
    return (
        f"Experiment: {session_folder_name} <br> "
        f"Test sensors: {sensors_text} | "
        f"Time window: {formatted_time}"
    )


if __name__ == "__main__":
    import os
    import click
    import pathlib

    @click.group()
    def cli():
        pass

    @cli.command()
    @click.argument("session_folder",
                    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True,
                                    path_type=pathlib.Path))
    @click.option("--unnormalize", is_flag=True, help="Plot in original space when experiment data is normalized.")
    def plot(session_folder: pathlib.Path, unnormalize: bool):
        session_data = load_training_session(str(session_folder))

        experiment_files = session_data['experiment_files']
        experiment_file = experiment_files[0]
        print(f"Plotting experiment {experiment_file.name}")
        experiment = np.load(experiment_file, allow_pickle=True)
        train_data = experiment['train_data']
        val_data = experiment['val_data']
        test_data = experiment['test_data']
        train_mask = experiment['train_mask']
        val_mask = experiment['val_mask']
        test_mask = experiment['test_mask']
        val_k_loss = experiment['val_k_loss']
        train_k_loss = experiment['train_k_loss']
        train_k_output = experiment['train_k_output']
        val_min_idx = experiment['val_min_idx']

        if unnormalize and session_data["configuration"].get("normalize", False):
            if "minmax_map" not in experiment:
                raise click.ClickException(
                    f"{experiment_file.name} does not contain minmax_map. Repair the session artifacts first."
                )

            pollutant_id = int(session_data["configuration"]["pollutants"][0])
            minmax_map = experiment["minmax_map"].item()
            if pollutant_id not in minmax_map:
                raise click.ClickException(f"Missing normalization stats for pollutant {pollutant_id}.")

            mean, std = minmax_map[pollutant_id]
            train_data = _denormalize_masked(train_data, train_mask, mean=mean, std=std)
            val_data = _denormalize_masked(val_data, val_mask, mean=mean, std=std)
            test_data = _denormalize_masked(test_data, test_mask, mean=mean, std=std)
            train_k_output = _denormalize_full(train_k_output, mean=mean, std=std)
            train_k_loss = _denormalize_losses(train_k_loss, std=std)
            val_k_loss = _denormalize_losses(val_k_loss, std=std)

        val_k_loss = val_k_loss[:, 0]         # (K, 1, T, 2) -> (K, T, 2)
        train_k_loss = train_k_loss[:, 0]      # (K, 1, T, 2) -> (K, T, 2)
        train_k_output = train_k_output[:, 0]  # (K, 1, T, H, W) -> (K, T, H, W)

        output = np.mean(train_k_output, axis=0)
        plot_data = {
            'y':  output,
            'train_loss': train_k_loss.mean(axis=0),
            'val_loss': val_k_loss.mean(axis=0)
        }

        plot_title = _build_plot_title(session_folder.name, experiment_file.name)

        all_data = (train_data + val_data + test_data)[0, 0, 0]
        limits = (np.nanmin(all_data), np.nanmax(all_data))

        video = plot_video(data=plot_data, title=plot_title, limits=limits)
        add_gt_points(video, train_data[0, 0, -1], train_mask[0, 0, -1], name="train")
        add_gt_points(video, val_data[0, 0, -1], val_mask[0, 0, -1], color="yellow", name="validation")
        add_gt_points(video, test_data[0, 0, -1], test_mask[0, 0, -1], color="blue", name="test")

        video_name = f"video.html" if not unnormalize else f"video_unnormalized.html"
        video.write_html(os.path.join(session_folder, video_name))

    @cli.command(name="plot-baseline")
    @click.argument("session_folder",
                    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True,
                                    path_type=pathlib.Path))
    @click.option("--method", type=click.Choice(["dip", "kriging", "idw"], case_sensitive=False), required=True)
    @click.option("--unnormalize", is_flag=True, help="Plot in original space when experiment data is normalized.")
    def plot_baseline(session_folder: pathlib.Path, method: str, unnormalize: bool):
        session_data = load_training_session(str(session_folder))

        experiment_files = session_data['experiment_files']
        experiment_file = experiment_files[0]
        print(f"Plotting {method} baseline for experiment {experiment_file.name}")
        experiment = np.load(experiment_file, allow_pickle=True)
        train_data = experiment['train_data']
        val_data = experiment['val_data']
        test_data = experiment['test_data']
        train_mask = experiment['train_mask']
        val_mask = experiment['val_mask']
        test_mask = experiment['test_mask']
        train_output = experiment['train_output']

        if unnormalize and session_data["configuration"].get("normalize", False):
            if "minmax_map" not in experiment:
                raise click.ClickException(
                    f"{experiment_file.name} does not contain minmax_map. Repair the session artifacts first."
                )

            pollutant_id = int(session_data["configuration"]["pollutants"][0])
            minmax_map = experiment["minmax_map"].item()
            if pollutant_id not in minmax_map:
                raise click.ClickException(f"Missing normalization stats for pollutant {pollutant_id}.")

            mean, std = minmax_map[pollutant_id]
            train_data = _denormalize_masked(train_data, train_mask, mean=mean, std=std)
            val_data = _denormalize_masked(val_data, val_mask, mean=mean, std=std)
            test_data = _denormalize_masked(test_data, test_mask, mean=mean, std=std)
            train_output = _denormalize_full(train_output, mean=mean, std=std)

        method_name = method.lower()
        if method_name == "dip":
            plot_data = _build_dip_plot_data(
                train_output=train_output,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
            )
        else:
            plot_data = _build_baseline_plot_data(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                method=method_name,
            )

        plot_title = _build_plot_title(session_folder.name, experiment_file.name) + f" | {method_name.upper()}"
        all_data = (train_data + val_data + test_data)[0, 0, 0]
        limits = (np.nanmin(all_data), np.nanmax(all_data))

        video = plot_video(data=plot_data, title=plot_title, limits=limits)
        add_gt_points(video, train_data[0, 0, -1], train_mask[0, 0, -1], name="train")
        add_gt_points(video, val_data[0, 0, -1], val_mask[0, 0, -1], color="yellow", name="validation")
        add_gt_points(video, test_data[0, 0, -1], test_mask[0, 0, -1], color="blue", name="test")

        video.write_html(f"{session_folder}/surface_{method_name}.html")

    cli()
