import networkx as nx
import pytest

import graph_coloring as gc
from graph_coloring import benchmarks


@pytest.mark.parametrize(
    "G, chi",
    [
        (nx.empty_graph(4), 1),
        (nx.path_graph(5), 2),
        (nx.cycle_graph(7), 3),
        (nx.petersen_graph(), 3),
        (nx.wheel_graph(6), 4),  # hub + odd rim C5
        (nx.complete_graph(6), 6),
        (benchmarks.load("myciel4")[0], 5),
        (benchmarks.load("queen6_6")[0], 7),
    ],
)
def test_chromatic_number(G, chi):
    assert gc.chromatic_number(G) == chi


def test_k_coloring_returns_proper_coloring():
    G = nx.erdos_renyi_graph(40, 0.2, seed=0)
    k = gc.chromatic_number(G)
    status, coloring = gc.k_coloring(G, k)
    assert status == "found"
    assert gc.is_proper(G, coloring)
    assert set(coloring.values()) <= set(range(k))
    assert gc.k_coloring(G, k - 1) == ("infeasible", None)


def test_k_coloring_edge_cases():
    assert gc.k_coloring(nx.Graph(), 3) == ("found", {})
    assert gc.k_coloring(nx.path_graph(2), 0) == ("infeasible", None)
    G = nx.Graph([(0, 0)])
    assert gc.k_coloring(G, 5) == ("infeasible", None)


def test_node_limit():
    G, _ = benchmarks.load("myciel5")
    assert gc.k_coloring(G, 5, node_limit=10) == ("limit", None)
    assert gc.chromatic_number(G, node_limit=10) is None


def test_estimate_uses_exact_search_to_close_the_gap():
    # Wheel with odd rim: clique bound 3, but chi = 4.
    G = nx.wheel_graph(8)  # hub + C7
    s = gc.estimate_chromatic_number(G, n_iter=10_000, seed=0)
    assert (s.lower, s.upper, s.exact) == (4, 4, True)
    s = gc.estimate_chromatic_number(G, n_iter=10_000, seed=0, exact_node_limit=0)
    assert (s.lower, s.upper, s.exact) == (3, 4, False)
