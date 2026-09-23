"""Frequency (channel) assignment for radio transmitters.

Transmitters closer than an interference radius ``r`` must use different
channels. The interference graph is a *unit disk graph* and the minimum
number of channels is its chromatic number.

Unit disk graphs are a nice test case because they are far from random:
any clique must fit in a region of diameter ``r``, cliques are large and
local, and ``chi`` is usually equal or very close to ``omega``. Greedy
coloring is a 3-approximation on them (Marathe et al., 1995).
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .maps import PALETTE


def transmitters(
    n: int = 150, radius: float = 0.12, clusters: int = 0, seed: int | None = None
) -> nx.Graph:
    """Random transmitters in the unit square and their interference graph.

    With ``clusters > 0`` the transmitters are concentrated around that many
    "city centres" (a Gaussian mixture), which mimics real networks with
    dense urban areas and sparse rural ones. Positions are stored in the
    ``"pos"`` node attribute.
    """
    rng = np.random.default_rng(seed)
    if clusters:
        centres = rng.uniform(0.15, 0.85, size=(clusters, 2))
        which = rng.integers(clusters, size=n)
        pts = centres[which] + rng.normal(scale=0.08, size=(n, 2))
        pts = np.clip(pts, 0, 1)
    else:
        pts = rng.uniform(size=(n, 2))
    pos = {i: (float(x), float(y)) for i, (x, y) in enumerate(pts)}
    return nx.random_geometric_graph(n, radius, pos=pos)


def plot_channels(
    G: nx.Graph,
    coloring: dict[int, int] | None = None,
    ax: plt.Axes | None = None,
    radius: float | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot transmitters colored by channel. If ``radius`` is given, draw
    each transmitter's interference disk (radius/2, so that two disks
    overlap exactly when the transmitters interfere)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    pos = nx.get_node_attributes(G, "pos")
    k = max(coloring.values()) + 1 if coloring else 1
    palette = PALETTE if k <= len(PALETTE) else [plt.cm.tab20(i % 20) for i in range(k)]
    colors = ["#999999" if coloring is None else palette[coloring[v]] for v in G.nodes]
    if radius is not None:
        for v, c in zip(G.nodes, colors, strict=True):
            ax.add_patch(plt.Circle(pos[v], radius / 2, color=c, alpha=0.15, lw=0))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.6)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=30, edgecolors="black", linewidths=0.4
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    return ax


def expected_degree(n: int, radius: float) -> float:
    """Approximate mean degree ``(n - 1) * pi * r^2`` (ignoring border effects)."""
    return (n - 1) * math.pi * radius**2
