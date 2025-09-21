import json
import os
from itertools import cycle
from typing import Dict, List, Optional

import einops
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.auto import tqdm


def plot_scores_plotly(
    scores_group: Dict[str, Float[Tensor, "n_samples n_layer"]],
    layer_of_interest,
    title,
    ylabel,
    artifact_dir,
    artifact_name,
    figsize=(10, 6),
    color_sequence=None,
):
    first_group_key = next(iter(scores_group))
    n_layer = scores_group[first_group_key].shape[1]
    layers = list(range(n_layer))

    fig = go.Figure()

    if color_sequence is None:
        # Using a few distinct Plotly default colors
        color_sequence = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
        ]
    color_iterator = cycle(color_sequence)

    for group_name, scores_tensor in scores_group.items():
        scores_np = scores_tensor.cpu().numpy()
        mean_scores = np.mean(scores_np, axis=0)
        min_scores = np.min(scores_np, axis=0)
        max_scores = np.max(scores_np, axis=0)

        current_color = next(color_iterator)

        if current_color.startswith("#") and len(current_color) == 7:
            r, g, b = (
                int(current_color[1:3], 16),
                int(current_color[3:5], 16),
                int(current_color[5:7], 16),
            )
            fill_color_rgba = f"rgba({r},{g},{b},0.1)"  # 10% opacity
        else:
            # Fallback for named colors, might require more complex mapping or just use a generic light grey
            # or rely on Plotly to convert named color to rgba internally if it supports it
            fill_color_rgba = f"rgba(192,192,192,0.1)"

        fig.add_trace(
            go.Scatter(
                x=layers,
                y=min_scores,
                mode="lines",
                line=dict(width=0),  # Invisible line
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=layers,
                y=max_scores,
                mode="lines",
                fill="tonexty",  # Fills the area between this trace and the previous one (min_scores)
                fillcolor=fill_color_rgba,
                line=dict(width=0),  # Invisible line
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Add bold mean line - this trace gets the legend entry
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=mean_scores,
                mode="lines",
                line=dict(color=current_color, width=3),  # Bold line with group color
                name=group_name,  # This creates the legend entry for this group
                hovertemplate=f"<b>Layer:</b> %{{x}}<br><b>{group_name} Mean:</b> %{{y:.3f}}<extra></extra>",
            )
        )

    if layer_of_interest is not None:
        fig.add_vline(
            x=layer_of_interest,
            line_dash="dot",
            line_color="grey",
            line_width=2,
            opacity=0.7,
        )
        fig.add_annotation(
            x=layer_of_interest,
            y=0.8,
            yref="paper",
            text=f"Layer {layer_of_interest}",
            showarrow=False,
            font=dict(color="grey", size=14),
            xshift=30,
            yanchor="middle",
        )

    fig.update_layout(
        title=dict(
            text=title, x=0.05, xanchor="left", font=dict(size=20, color="black")
        ),
        xaxis_title=dict(text="Layer", font=dict(size=16, color="black")),
        yaxis_title=dict(text=ylabel, font=dict(size=16, color="black")),
        xaxis=dict(
            zeroline=False,
            linewidth=1,
            ticks="outside",
        ),
        yaxis=dict(
            zeroline=False,
            linewidth=1,
            ticks="outside",
        ),
        margin=dict(l=80, r=80, t=100, b=120),
        showlegend=True,  # Explicitly enable legend
        width=figsize[0] * 100,
        height=figsize[1] * 100,
    )

    output_dir = os.path.join(artifact_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path_pdf = os.path.join(output_dir, f"{artifact_name}.pdf")

    fig.write_image(output_path_pdf, format="pdf")
    fig.write_image(output_path_pdf, format="pdf")
    print(f"Plot saved to {output_path_pdf}")

    fig.show()


def plot_refusal_scores_plotly(
    refusal_scores,
    baseline_refusal_score,
    token_labels,
    title: str,
    ylabel: str,
    artifact_dir: str,
    artifact_name: str,
    layers: None,
    figsize=(10, 6),
    show=True,
):
    n_pos, n_layer = refusal_scores.shape

    fig = go.Figure()

    for i in range(-n_pos, 0):
        y_data = refusal_scores[i].cpu().numpy()
        fig.add_trace(
            go.Scatter(
                x=list(range(n_layer)),
                y=y_data,
                mode="lines",
                name=f"{i}: {repr(token_labels[i])}",
                hovertemplate="<b>Layer:</b> %{x}<br><b>Score:</b> %{y:.2f}<extra></extra>",
            )
        )

    if baseline_refusal_score is not None:
        fig.add_hline(
            y=baseline_refusal_score,
            line_dash="dash",
            line_color="black",
            annotation_text="Baseline",
            annotation_position="bottom right",
            annotation_font_color="black",
            annotation_borderpad=0,
        )

    if layers is not None:
        for layer in layers:
            fig.add_vline(
                x=layer,
                line_dash="dash",
                line_color="grey",
            )
            fig.add_annotation(
                x=layer,
                y=0.8,
                yref="paper",  # yref="paper" positions relative to plot height (0 to 1)
                text=f"Layer {layer}",
                showarrow=False,
                font=dict(color="grey", size=14),
                xshift=30,  # Shift text slightly to the right of the line
                yanchor="middle",  # Center text vertically on its y-position
            )

    fig.update_layout(
        title_text=title,
        xaxis_title_text="Layer source of direction (resid_pre)",
        yaxis_title_text=ylabel,
        legend_title_text="Position source of direction",  # Title for the legend
        width=figsize[0] * 100,  # Convert width from inches to pixels
        height=figsize[1] * 100,  # Convert height from inches to pixels
    )

    # Ensure the output directory exists before saving the plot
    output_dir = os.path.join(artifact_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Define the full path for the output PDF file
    output_path = os.path.join(output_dir, f"{artifact_name}.pdf")

    fig.write_image(output_path, format="pdf")
    fig.write_image(output_path, format="pdf")
    print(f"Plot saved to {output_path}")

    if show:
        fig.show()


def plot_refusal_scores(
    refusal_scores: Float[Tensor, "n_pos n_layer"],
    baseline_refusal_score: Optional[float],
    token_labels: List[str],
    title: str,
    ylabel: str,
    artifact_dir: str,
    artifact_name: str,
    legend_loc: str = "lower left",
    figsize=(10, 6),
):
    n_pos, n_layer = refusal_scores.shape

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=figsize)  # width and height in inches

    # Add a trace for each position to extract
    for i in range(-n_pos, 0):
        ax.plot(
            list(range(n_layer)),
            refusal_scores[i].cpu().numpy(),
            label=f"{i}: {repr(token_labels[i])}",
        )

    if baseline_refusal_score is not None:
        # Add a horizontal line for the baseline
        ax.axhline(y=baseline_refusal_score, color="black", linestyle="--")
        ax.annotate(
            "Baseline",
            xy=(1, baseline_refusal_score),
            xytext=(8, 10),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            horizontalalignment="right",
            verticalalignment="center",
        )

    ax.set_title(title)
    ax.set_xlabel("Layer source of direction (resid_pre)")
    ax.set_ylabel(ylabel)
    ax.legend(title="Position source of direction", loc=legend_loc)

    plt.savefig(f"{artifact_dir}/figures/{artifact_name}.pdf")
