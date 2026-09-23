"""Benchmark graphs with known chromatic numbers.

These reproduce instances from the DIMACS graph coloring challenge
(Johnson & Trick, 1996) that are defined constructively, so they can be
generated instead of downloaded.
"""

from __future__ import annotations

import networkx as nx


def queen_graph(rows: int, cols: int | None = None) -> nx.Graph:
    """DIMACS ``queen{rows}_{cols}``: one node per square, adjacent when a
    queen on one square attacks the other (same row, column or diagonal)."""
    cols = rows if cols is None else cols
    G = nx.Graph()
    squares = [(r, c) for r in range(rows) for c in range(cols)]
    G.add_nodes_from(squares)
    for i, (r1, c1) in enumerate(squares):
        for r2, c2 in squares[i + 1 :]:
            if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                G.add_edge((r1, c1), (r2, c2))
    return G


def mycielski(k: int) -> nx.Graph:
    """DIMACS ``myciel{k}``: triangle-free graph with chromatic number
    ``k + 1`` (``myciel3`` is the Grötzsch graph, ``chi = 4``).

    The clique bound is useless here (``omega = 2``), which makes these
    graphs a classic stress test for coloring heuristics.
    """
    # networkx's mycielski_graph(n) has chromatic number n, and DIMACS
    # myciel{k} has chromatic number k + 1.
    return nx.mycielski_graph(k + 1)


# name -> (constructor, known chromatic number)
KNOWN = {
    "myciel3": (lambda: mycielski(3), 4),
    "myciel4": (lambda: mycielski(4), 5),
    "myciel5": (lambda: mycielski(5), 6),
    "myciel6": (lambda: mycielski(6), 7),
    "queen5_5": (lambda: queen_graph(5), 5),
    "queen6_6": (lambda: queen_graph(6), 7),
    "queen7_7": (lambda: queen_graph(7), 7),
    "queen8_8": (lambda: queen_graph(8), 9),
    "queen8_12": (lambda: queen_graph(8, 12), 12),
}


def load(name: str) -> tuple[nx.Graph, int]:
    """Return ``(graph, chromatic_number)`` for a benchmark in :data:`KNOWN`."""
    make, chi = KNOWN[name]
    return make(), chi
