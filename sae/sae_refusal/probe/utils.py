import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
from jaxtyping import Float
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve
from tqdm.auto import tqdm
from transformers import GenerationConfig

from sae_refusal import clear_memory
from sae_refusal.model import ModelBase
from sae_refusal.pipeline.hook import add_hooks


def np_sigmoid(x):
    return np.where(
        x >= 0,  # condition
        1 / (1 + np.exp(-x)),  # For positive values
        np.exp(x) / (1 + np.exp(x)),  # For negative values
    )


def plot_roc_curve(labels, scores, artifact_dir, title="Probe Performance"):
    junk_scores = scores[labels == 1]
    non_junk_scores = scores[labels == 0]

    # Calculate ROC curve data
    probabilities = np_sigmoid(scores)
    fpr, tpr, _ = roc_curve(labels, probabilities)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("(a) ROC Curve", "(b) Junk Detector Activation Distribution"),
    )

    # --- Plot 1: ROC Curve on a Log-Log Scale ---
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="ROC Curve",
            line=dict(color="orange", width=2),
        ),
        row=1,
        col=1,
    )

    # --- Plot 2: Overlaid Histograms ---
    fig.add_trace(
        go.Histogram(
            x=non_junk_scores,
            name="Non-Junk (Harmless)",
            marker_color="#3366CC",
            nbinsx=50,
            opacity=0.7,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Histogram(
            x=junk_scores,
            name="Junk (Hazardous)",
            marker_color="#DC3912",
            nbinsx=50,
            opacity=0.7,
        ),
        row=1,
        col=2,
    )

    # --- Styling and Layout ---
    # Update axes for ROC plot
    fig.update_xaxes(title_text="False Positive Rate (FPR)", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate (TPR)", row=1, col=1)

    # Update axes for Histogram plot
    fig.update_xaxes(title_text="Detector Activation (Linear Score)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    # Update overall layout
    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_x=0.5,
        barmode="overlay",  # This makes the histograms overlap
        legend_title_text="Class",
    )

    fig.show()

    if artifact_dir is not None:
        fig.write_image(artifact_dir, width=1000, height=500, format="pdf")

def plot_precision_recall_curve(labels, scores, artifact_dir, title="Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(labels, scores)
    avg_precision = average_precision_score(labels, scores)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'PR Curve (AP = {avg_precision:.3f})',
        line=dict(color='purple', width=2)
    ))

    fig.update_layout(
        title=f"<b>{title}</b>",
        title_x=0.5,
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[0, 1.05]),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.01, y=0.01, xanchor='left', yanchor='bottom')
    )
    fig.show()

    if artifact_dir is not None:
        fig.write_image(artifact_dir, width=1000, height=500, format="pdf")