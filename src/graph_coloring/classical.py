"""Classical constructive coloring heuristics.

These always return a *proper* coloring; the question is how many colors
they use. They serve as baselines for simulated annealing and provide
upper bounds on the chromatic number.
"""

from __future__ import annotations

import heapq
from collections.abc import Hashable, Iterable

import networkx as nx


def greedy(G: nx.Graph, order: Iterable[Hashable] | None = None) -> dict[Hashable, int]:
    """First-fit coloring: visit nodes in ``order`` and give each the
    smallest color not used by an already-colored neighbour.

    Uses at most ``max_degree + 1`` colors for any order.
    """
    coloring: dict[Hashable, int] = {}
    for v in G.nodes if order is None else order:
        used = {coloring[w] for w in G.adj[v] if w in coloring}
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    return coloring


def welsh_powell(G: nx.Graph) -> dict[Hashable, int]:
    """Greedy coloring with nodes sorted by decreasing degree
    (Welsh & Powell, 1967). Uses at most ``max_i min(d_i + 1, i)`` colors,
    where ``d_1 >= d_2 >= ...`` are the sorted degrees.
    """
    order = sorted(G.nodes, key=G.degree, reverse=True)
    return greedy(G, order)


def smallest_last(G: nx.Graph) -> dict[Hashable, int]:
    """Greedy coloring in smallest-last order (Matula & Beck, 1983).

    Uses at most ``degeneracy(G) + 1`` colors.
    """
    H = G.copy()
    order = []
    while H:
        v = min(H.nodes, key=H.degree)
        order.append(v)
        H.remove_node(v)
    return greedy(G, reversed(order))


def dsatur(G: nx.Graph) -> dict[Hashable, int]:
    """DSATUR heuristic (Brélaz, 1979).

    Repeatedly colors the uncolored node with the most distinct colors in
    its neighbourhood (its *saturation*), breaking ties by degree, with the
    smallest available color. Exact on bipartite graphs, cycles and wheels.
    """
    coloring: dict[Hashable, int] = {}
    neighbour_colors: dict[Hashable, set[int]] = {v: set() for v in G.nodes}
    tie = {v: i for i, v in enumerate(G.nodes)}
    heap = [(0, -G.degree(v), tie[v], v) for v in G.nodes]
    heapq.heapify(heap)
    while heap:
        neg_sat, _, _, v = heapq.heappop(heap)
        if v in coloring or -neg_sat != len(neighbour_colors[v]):
            continue  # stale entry
        used = neighbour_colors[v]
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
        for w in G.adj[v]:
            if w not in coloring and c not in neighbour_colors[w]:
                neighbour_colors[w].add(c)
                heapq.heappush(heap, (-len(neighbour_colors[w]), -G.degree(w), tie[w], w))
    return coloring


HEURISTICS = {
    "greedy": greedy,
    "welsh_powell": welsh_powell,
    "smallest_last": smallest_last,
    "dsatur": dsatur,
}
