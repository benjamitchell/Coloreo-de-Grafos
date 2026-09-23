"""Cost function for the graph coloring problem.

A coloring is a map ``x: V -> {0, ..., k-1}``. Its cost is the number of
*conflicting* edges, i.e. edges whose endpoints share a color:

    H(x) = sum_{u ~ v} 1[x_u == x_v]

A coloring is *proper* if and only if ``H(x) == 0``.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping

import networkx as nx

Coloring = Mapping[Hashable, int]


def conflicts(G: nx.Graph, coloring: Coloring) -> int:
    """Return the number of edges whose endpoints share a color."""
    return sum(1 for u, v in G.edges() if coloring[u] == coloring[v])


def conflicting_edges(G: nx.Graph, coloring: Coloring) -> list[tuple[Hashable, Hashable]]:
    """Return the list of edges whose endpoints share a color."""
    return [(u, v) for u, v in G.edges() if coloring[u] == coloring[v]]


def is_proper(G: nx.Graph, coloring: Coloring) -> bool:
    """Return ``True`` if no edge joins two nodes of the same color."""
    return all(coloring[u] != coloring[v] for u, v in G.edges())


def num_colors(coloring: Coloring) -> int:
    """Return the number of distinct colors used by ``coloring``."""
    return len(set(coloring.values()))


def delta_recolor(G: nx.Graph, coloring: Coloring, node: Hashable, new_color: int) -> int:
    """Return ``H(x') - H(x)`` where ``x'`` recolors ``node`` to ``new_color``.

    Only the edges incident to ``node`` can change, so this costs
    ``O(deg(node))`` instead of ``O(|E|)``.
    """
    old_color = coloring[node]
    if new_color == old_color:
        return 0
    gained = lost = 0
    for nb in G.adj[node]:
        c = coloring[nb]
        if c == new_color:
            gained += 1
        elif c == old_color:
            lost += 1
    return gained - lost
