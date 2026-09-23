"""Exact k-colorability by backtracking.

Deciding whether ``chi(G) <= k`` is NP-complete, but for the graphs in this
project (maps, small benchmarks) a DSATUR-ordered backtracking search with
forward checking settles it quickly. It complements simulated annealing:
annealing finds colorings fast but can never prove that none exists; this
search can, as long as the graph is small enough.

The search always branches on the uncolored node with the fewest remaining
colors (ties broken by degree), removes the chosen color from its
neighbours' domains, and backtracks as soon as some domain becomes empty.
Color symmetry is broken by never opening more than one new color at a time.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Literal

import networkx as nx

Status = Literal["found", "infeasible", "limit"]


class _NodeLimit(Exception):
    pass


def k_coloring(
    G: nx.Graph, k: int, node_limit: int | None = 1_000_000
) -> tuple[Status, dict[Hashable, int] | None]:
    """Decide whether ``G`` has a proper ``k``-coloring.

    Returns ``("found", coloring)``, ``("infeasible", None)`` when no
    ``k``-coloring exists (a proof that ``chi(G) > k``), or
    ``("limit", None)`` if more than ``node_limit`` search nodes were needed.
    """
    nodes = list(G.nodes)
    n = len(nodes)
    if n == 0:
        return "found", {}
    if k < 1:
        return "infeasible", None
    index = {v: i for i, v in enumerate(nodes)}
    adj = [[index[w] for w in G.adj[v] if w != v] for v in nodes]
    if any(v in G.adj[v] for v in nodes):
        return "infeasible", None  # a self-loop can never be properly colored
    degree = [len(a) for a in adj]
    full = (1 << k) - 1
    domain = [full] * n  # bitmask of colors still allowed for each node
    color = [-1] * n
    counter = [0]

    def choose() -> int:
        best, best_key = -1, None
        for v in range(n):
            if color[v] < 0:
                key = (domain[v].bit_count(), -degree[v])
                if best_key is None or key < best_key:
                    best, best_key = v, key
        return best

    def search(colored: int, used: int) -> bool:
        if colored == n:
            return True
        counter[0] += 1
        if node_limit is not None and counter[0] > node_limit:
            raise _NodeLimit
        v = choose()
        # colors 0..used-1 are interchangeable only up to the first new one
        options = domain[v] & ((1 << min(used + 1, k)) - 1)
        while options:
            bit = options & -options
            options ^= bit
            c = bit.bit_length() - 1
            changed = []
            ok = True
            for w in adj[v]:
                if color[w] < 0 and domain[w] & bit:
                    domain[w] ^= bit
                    changed.append(w)
                    if not domain[w]:
                        ok = False
                        break
            if ok:
                color[v] = c
                if search(colored + 1, max(used, c + 1)):
                    return True
                color[v] = -1
            for w in changed:
                domain[w] |= bit
        return False

    try:
        found = search(0, 0)
    except _NodeLimit:
        return "limit", None
    except RecursionError:  # pragma: no cover - only for enormous graphs
        return "limit", None
    if not found:
        return "infeasible", None
    return "found", {v: color[i] for i, v in enumerate(nodes)}


def chromatic_number(G: nx.Graph, node_limit: int | None = 1_000_000) -> int | None:
    """Exact chromatic number, or ``None`` if the search hits ``node_limit``."""
    from .classical import dsatur

    coloring = dsatur(G)
    upper = len(set(coloring.values()))
    for k in range(upper - 1, 0, -1):
        status, _ = k_coloring(G, k, node_limit)
        if status == "infeasible":
            return k + 1
        if status == "limit":
            return None
    return upper
