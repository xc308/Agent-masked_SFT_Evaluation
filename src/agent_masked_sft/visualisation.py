from typing import List, Dict, Tuple
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np


def compute_truth_preference_deltas(
    score_rows: List[Dict],
) -> Tuple[List[float], List[float]]:
    """
    Compute truth preference deltas for standard and interventional SFT.

    delta = average log p(true completion) - average log p(false completion)

    Higher delta means stronger preference for the true statement.
    Positive delta means the model prefers the true completion.
    Negative delta means the model prefers the false completion.
    """
    delta_std = [
        row["score_std"]["true"] - row["score_std"]["false"]
        for row in score_rows
    ]

    delta_iv = [
        row["score_iv"]["true"] - row["score_iv"]["false"]
        for row in score_rows
    ]

    return delta_std, delta_iv


def plot_truth_preference_deltas(
    score_rows: List[Dict],
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """
    Plot truth preference deltas for standard and interventional SFT.

    Args:
        score_rows:
            Evaluation rows containing score_std and score_iv.

        output_path:
            If provided, save the figure to this path.

        show:
            If True, display the plot interactively using plt.show().
            In scripts, saving is usually more reliable than showing.
    """
    delta_std, delta_iv = compute_truth_preference_deltas(score_rows)

    fig, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(score_rows))
    width = 0.4

    ax.bar(
        x - width / 2,
        delta_std,
        width,
        label="Std SFT",
    )

    ax.bar(
        x + width / 2,
        delta_iv,
        width,
        label="IV SFT",
    )

    ax.axhline(
        0,
        color="k",
        linewidth=0.5,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [row["fact"]["topic"][:18] for row in score_rows],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        r"$\log p(\mathrm{true}) - \log p(\mathrm{false})$ (per token)"
    )

    ax.set_title(
        "Preference for truth over the agent's false statement "
        "(higher is better)"
    )

    ax.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved truth preference plot to {output_path}")

    if show:
        plt.show()

    plt.close(fig)
