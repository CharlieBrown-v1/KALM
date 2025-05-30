# pylint: disable=redefined-builtin,exec-used
# type: ignore

import glob
import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import click
import gym
import numpy as np

from ._version import __version__
from .algos import (
    QLearningAlgoBase,
    StatefulTransformerWrapper,
    TransformerAlgoBase,
)
from .base import load_learnable
from .envs import Monitor
from .metrics.utility import (
    evaluate_qlearning_with_environment,
    evaluate_transformer_with_environment,
)

if TYPE_CHECKING:
    import matplotlib.pyplot


def print_stats(path: str) -> None:
    data = np.loadtxt(path, delimiter=",")
    print("FILE NAME  : ", path)
    print("EPOCH      : ", data[-1, 0])
    print("TOTAL STEPS: ", data[-1, 1])
    print("MAX VALUE  : ", np.max(data[:, 2]))
    print("MIN VALUE  : ", np.min(data[:, 2]))
    print("STD VALUE  : ", np.std(data[:, 2]))


def get_plt() -> "matplotlib.pyplot":
    import matplotlib.pyplot as plt

    try:
        # enable seaborn style if available
        import seaborn as sns

        sns.set()
    except ImportError:
        pass
    return plt


def _compute_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    assert values.ndim == 1
    results: List[float] = []
    # average over past data
    for i in range(values.shape[0]):
        start = max(0, i - window)
        results.append(float(np.mean(values[start : i + 1])))
    return np.array(results)


@click.group()
def cli() -> None:
    print(f"d3rlpy command line interface (Version {__version__})")


@cli.command(short_help="Show statistics of save metrics.")
@click.argument("path")
def stats(path: str) -> None:
    print_stats(path)


@cli.command(short_help="Plot saved metrics (requires matplotlib).")
@click.argument("path", nargs=-1)
@click.option(
    "--window", default=1, show_default=True, help="moving average window."
)
@click.option("--show-steps", is_flag=True, help="use iterations on x-axis.")
@click.option("--show-max", is_flag=True, help="show maximum value.")
@click.option("--label", multiple=True, help="label in legend.")
@click.option("--xlim", nargs=2, type=float, help="limit on x-axis (tuple).")
@click.option("--ylim", nargs=2, type=float, help="limit on y-axis (tuple).")
@click.option("--title", help="title of the plot.")
@click.option("--ylabel", default="value", help="label on y-axis.")
@click.option("--save", help="flag to save the plot as an image.")
def plot(
    path: List[str],
    window: int,
    show_steps: bool,
    show_max: bool,
    label: Optional[Sequence[str]],
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
    title: Optional[str],
    ylabel: str,
    save: str,
) -> None:
    plt = get_plt()

    max_y_values = []
    min_x_values = []
    max_x_values = []

    if label:
        assert len(label) == len(
            path
        ), "--labels must be provided as many as the number of paths"

    for i, p in enumerate(path):
        data = np.loadtxt(p, delimiter=",")

        # filter to smooth data
        y_data = _compute_moving_average(data[:, 2], window)

        # create label
        if label:
            _label = label[i]
        elif len(p.split(os.sep)) > 1:
            _label = "/".join(p.split(os.sep)[-2:])
        else:
            _label = p

        if show_steps:
            x_data = data[:, 1]
        else:
            x_data = data[:, 0]

        max_y_values.append(np.max(data[:, 2]))
        min_x_values.append(np.min(x_data))
        max_x_values.append(np.max(x_data))

        # show statistics
        print("")
        print_stats(p)

        plt.plot(x_data, y_data, label=_label)

    if show_max:
        plt.plot(
            [np.min(min_x_values), np.max(max_x_values)],
            [np.max(max_y_values), np.max(max_y_values)],
            color="black",
            linestyle="dashed",
        )

    plt.xlabel("steps" if show_steps else "epochs")
    plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])

    if title:
        plt.title(title)

    plt.legend()
    if save:
        plt.savefig(save)
    else:
        plt.show()


@cli.command(short_help="Plot saved metrics in a grid (requires matplotlib).")
@click.argument("path")
@click.option("--title", help="title of the plot.")
@click.option("--save", help="flag to save the plot as an image.")
def plot_all(
    path: str,
    title: Optional[str],
    save: str,
) -> None:
    plt = get_plt()

    # print params.json
    if os.path.exists(os.path.join(path, "params.json")):
        with open(os.path.join(path, "params.json"), "r") as f:
            params = json.loads(f.read())
        print("")
        for k, v in params.items():
            print(f"{k}={v}")

    metrics_names = sorted(list(glob.glob(os.path.join(path, "*.csv"))))
    n_cols = int(np.ceil(len(metrics_names) ** 0.5))
    n_rows = int(np.ceil(len(metrics_names) / n_cols))

    plt.figure(figsize=(12, 7))

    for i in range(n_rows):
        for j in range(n_cols):
            index = j + n_cols * i
            if index >= len(metrics_names):
                break

            plt.subplot(n_rows, n_cols, index + 1)

            data = np.loadtxt(metrics_names[index], delimiter=",")

            plt.plot(data[:, 0], data[:, 2])
            plt.title(os.path.basename(metrics_names[index]))
            plt.xlabel("epoch")
            plt.ylabel("value")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    if save:
        plt.savefig(save)
    else:
        plt.show()


@cli.command(
    short_help="Export saved model as inference model format (ONNX or TorchScript)."
)
@click.argument("model_path")
@click.argument("output_path")
def export(model_path: str, output_path: str) -> None:
    # load saved model
    print(f"Loading {model_path}...")
    algo = load_learnable(model_path)
    assert isinstance(
        algo, QLearningAlgoBase
    ), "Currently, only Q-learning algorithms are supported."

    # export inference model
    print(f"Exporting to {output_path}...")
    algo.save_policy(output_path)


def _exec_to_create_env(code: str) -> gym.Env[Any, Any]:
    print(f"Executing '{code}'")
    variables: Dict[str, Any] = {}
    exec(code, globals(), variables)
    if "env" not in variables:
        raise RuntimeError("env must be defined in env_header.")
    return variables["env"]  # type: ignore


@cli.command(short_help="Record episodes with the saved model.")
@click.argument("model_path")
@click.option("--env-id", default=None, help="Gym environment id.")
@click.option(
    "--env-header", default=None, help="one-liner to create environment."
)
@click.option("--out", default="videos", help="output directory path.")
@click.option(
    "--n-episodes", default=3, help="the number of episodes to record."
)
@click.option("--frame-rate", default=60, help="video frame rate.")
@click.option("--record-rate", default=1, help="record frame rate.")
@click.option(
    "--target-return",
    default=None,
    help="the target return for Decision Transformer variants.",
)
def record(
    model_path: str,
    env_id: Optional[str],
    env_header: Optional[str],
    out: str,
    n_episodes: int,
    frame_rate: float,
    record_rate: int,
    target_return: Optional[float],
) -> None:
    # load saved model
    print(f"Loading {model_path}...")
    algo = load_learnable(model_path)

    # wrap environment with Monitor
    env: gym.Env[Any, Any]
    if env_id is not None:
        env = gym.make(env_id)
    elif env_header is not None:
        env = _exec_to_create_env(env_header)
    else:
        raise ValueError("env_id or env_header must be provided.")

    wrapped_env = Monitor(
        env,
        out,
        video_callable=lambda ep: ep % 1 == 0,
        frame_rate=float(frame_rate),
        record_rate=int(record_rate),
    )

    # run episodes
    if isinstance(algo, QLearningAlgoBase):
        evaluate_qlearning_with_environment(
            algo, wrapped_env, n_episodes, render=True
        )
    elif isinstance(algo, TransformerAlgoBase):
        assert target_return is not None, "--target-return must be specified."
        evaluate_transformer_with_environment(
            StatefulTransformerWrapper(algo, float(target_return)),
            wrapped_env,
            n_episodes,
            render=True,
        )
    else:
        raise ValueError("invalid algo type.")


@cli.command(short_help="Run evaluation episodes with rendering.")
@click.argument("model_path")
@click.option("--env-id", default=None, help="Gym environment id.")
@click.option(
    "--env-header", default=None, help="one-liner to create environment."
)
@click.option("--n-episodes", default=3, help="the number of episodes to run.")
@click.option(
    "--target-return",
    default=None,
    help="the target return for Decision Transformer variants.",
)
def play(
    model_path: str,
    env_id: Optional[str],
    env_header: Optional[str],
    n_episodes: int,
    target_return: Optional[float],
) -> None:
    # load saved model
    print(f"Loading {model_path}...")
    algo = load_learnable(model_path)

    # wrap environment with Monitor
    env: gym.Env[Any, Any]
    if env_id is not None:
        env = gym.make(env_id)
    elif env_header is not None:
        env = _exec_to_create_env(env_header)
    else:
        raise ValueError("env_id or env_header must be provided.")

    # run episodes
    if isinstance(algo, QLearningAlgoBase):
        evaluate_qlearning_with_environment(algo, env, n_episodes, render=True)
    elif isinstance(algo, TransformerAlgoBase):
        assert target_return is not None, "--target-return must be specified."
        evaluate_transformer_with_environment(
            StatefulTransformerWrapper(algo, float(target_return)),
            env,
            n_episodes,
            render=True,
        )
    else:
        raise ValueError("invalid algo type.")
