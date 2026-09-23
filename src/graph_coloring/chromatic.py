"""Bounds on, and a search for, the chromatic number ``chi(G)``.

Lower bounds:

* ``omega(G) <= chi(G)``: a clique of size ``w`` needs ``w`` colors.

Upper bounds:

* ``chi(G) <= Delta(G) + 1`` (greedy in any order).
* Brooks (1941): ``chi(G) <= Delta(G)`` unless ``G`` has a connected
  component that is a complete graph or an odd cycle.
* ``chi(G) <= degeneracy(G) + 1`` (smallest-last greedy).
* Any proper coloring found by a heuristic.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

import networkx as nx

from .annealing import AnnealingResult, simulated_annealing
from .classical import dsatur
from .cost import num_colors
from .exact import k_coloring


def max_degree_bound(G: nx.Graph) -> int:
    """``Delta(G) + 1``, the bound used in the original project."""
    return max((d for _, d in G.degree), default=-1) + 1


def brooks_bound(G: nx.Graph) -> int:
    """Upper bound from Brooks' theorem, applied to each component.

    A connected graph that is neither complete nor an odd cycle satisfies
    ``chi <= Delta``; ``K_n`` needs ``n`` colors and odd cycles need 3.
    """
    bound = 0
    for comp in nx.connected_components(G):
        H = G.subgraph(comp)
        n, m = H.number_of_nodes(), H.number_of_edges()
        if m == n * (n - 1) // 2:  # complete graph K_n
            b = n
        elif n % 2 == 1 and all(d == 2 for _, d in H.degree):  # odd cycle
            b = 3
        else:
            b = max(d for _, d in H.degree)
        bound = max(bound, b)
    return bound


def degeneracy_bound(G: nx.Graph) -> int:
    """``degeneracy(G) + 1`` (the max core number plus one)."""
    if G.number_of_nodes() == 0:
        return 0
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return max(nx.core_number(G).values(), default=0) + 1


def clique_lower_bound(G: nx.Graph, exact_limit: int = 200) -> int:
    """Size of a (maximum, if ``G`` is small enough) clique.

    For graphs with more than ``exact_limit`` nodes a greedy clique is used,
    which is still a valid lower bound.
    """
    if G.number_of_nodes() == 0:
        return 0
    if G.number_of_nodes() <= exact_limit:
        clique, _ = nx.max_weight_clique(G, weight=None)
        return len(clique)
    return len(_greedy_clique(G))


def _greedy_clique(G: nx.Graph) -> list[Hashable]:
    best: list[Hashable] = []
    for start in sorted(G.nodes, key=G.degree, reverse=True)[:50]:
        clique = [start]
        cand = set(G.adj[start])
        while cand:
            v = max(cand, key=lambda u: len(cand & set(G.adj[u])))
            clique.append(v)
            cand &= set(G.adj[v])
        if len(clique) > len(best):
            best = clique
    return best


@dataclass
class ChromaticSearch:
    """Result of :func:`estimate_chromatic_number`."""

    lower: int
    """Proven lower bound: the clique number, raised by the exact search
    when it proves that no ``(upper - 1)``-coloring exists."""
    upper: int
    """Smallest ``k`` for which a proper ``k``-coloring was found."""
    coloring: dict[Hashable, int]
    """A proper coloring with ``upper`` colors."""
    runs: dict[int, AnnealingResult] = field(default_factory=dict, repr=False)
    """Annealing run for each ``k`` tried."""

    @property
    def exact(self) -> bool:
        """``True`` when the bounds meet, so ``chi(G) = upper`` is proven."""
        return self.lower == self.upper


def estimate_chromatic_number(
    G: nx.Graph,
    n_iter: int = 200_000,
    restarts: int = 3,
    seed: int | None = None,
    exact_node_limit: int | None = 200_000,
    **annealing_kwargs,
) -> ChromaticSearch:
    """Squeeze ``chi(G)`` between a clique lower bound and annealing.

    Start from the DSATUR coloring (a proven upper bound) and try to find a
    proper ``(k-1)``-coloring by simulated annealing, decreasing ``k`` while
    it succeeds. Failing to find a coloring does *not* prove none exists, so
    if the bounds still differ an exact backtracking search
    (:func:`graph_coloring.exact.k_coloring`) tries to settle ``k - 1``
    within ``exact_node_limit`` search nodes (``0`` disables it). Unless
    ``lower == upper`` at the end, ``upper`` is only an upper bound.
    """
    lower = clique_lower_bound(G)
    coloring = dsatur(G)
    upper = num_colors(coloring) if coloring else 0
    runs: dict[int, AnnealingResult] = {}

    k = upper - 1
    while k >= max(lower, 1):
        result = None
        for r in range(restarts):
            s = None if seed is None else seed + 1000 * k + r
            result = simulated_annealing(G, k, n_iter, seed=s, **annealing_kwargs)
            if result.solved:
                break
        runs[k] = result
        if not result.solved:
            break
        upper, coloring = k, result.coloring
        k -= 1

    while exact_node_limit != 0 and lower < upper:
        status, exact_coloring = k_coloring(G, upper - 1, exact_node_limit)
        if status == "infeasible":
            lower = upper
        elif status == "found":
            upper, coloring = upper - 1, exact_coloring
            continue
        break

    return ChromaticSearch(lower=lower, upper=upper, coloring=coloring, runs=runs)
