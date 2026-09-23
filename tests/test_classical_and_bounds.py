import networkx as nx
import pytest

import graph_coloring as gc
from graph_coloring import benchmarks

GRAPHS = {
    "petersen": nx.petersen_graph(),
    "gnp": nx.erdos_renyi_graph(80, 0.2, seed=0),
    "grid": nx.grid_2d_graph(6, 7),
    "wheel": nx.wheel_graph(8),
    "disconnected": nx.disjoint_union(nx.complete_graph(4), nx.cycle_graph(7)),
}


@pytest.mark.parametrize("name", list(gc.HEURISTICS))
@pytest.mark.parametrize("graph", list(GRAPHS))
def test_heuristics_give_proper_colorings_within_bounds(name, graph):
    G = GRAPHS[graph]
    coloring = gc.HEURISTICS[name](G)
    assert set(coloring) == set(G)
    assert gc.is_proper(G, coloring)
    assert gc.num_colors(coloring) <= gc.max_degree_bound(G)
    assert gc.clique_lower_bound(G) <= gc.num_colors(coloring)


def test_smallest_last_respects_degeneracy_bound():
    for G in GRAPHS.values():
        assert gc.num_colors(gc.smallest_last(G)) <= gc.degeneracy_bound(G)


def test_dsatur_is_exact_on_bipartite_and_cycles():
    assert gc.num_colors(gc.dsatur(nx.grid_2d_graph(5, 5))) == 2
    assert gc.num_colors(gc.dsatur(nx.cycle_graph(9))) == 3
    assert gc.num_colors(gc.dsatur(nx.wheel_graph(6))) == 4  # hub + odd rim C5


def test_brooks_bound():
    assert gc.brooks_bound(nx.complete_graph(5)) == 5
    assert gc.brooks_bound(nx.cycle_graph(7)) == 3
    assert gc.brooks_bound(nx.cycle_graph(8)) == 2
    assert gc.brooks_bound(nx.petersen_graph()) == 3
    assert gc.brooks_bound(GRAPHS["disconnected"]) == 4
    assert gc.brooks_bound(nx.empty_graph(3)) == 1


def test_estimate_chromatic_number_small_graphs():
    s = gc.estimate_chromatic_number(nx.petersen_graph(), n_iter=20_000, seed=0)
    assert (s.lower, s.upper, s.exact) == (3, 3, True)  # clique bound 2, closed by exact search
    assert gc.is_proper(nx.petersen_graph(), s.coloring)
    s = gc.estimate_chromatic_number(nx.complete_graph(7), n_iter=1_000, seed=0)
    assert s.exact and s.upper == 7


@pytest.mark.parametrize(
    "name, nodes, edges",
    [("myciel3", 11, 20), ("myciel5", 47, 236), ("queen5_5", 25, 160), ("queen8_8", 64, 728)],
)
def test_benchmarks_match_dimacs_sizes(name, nodes, edges):
    G, _ = benchmarks.load(name)
    assert (G.number_of_nodes(), G.number_of_edges()) == (nodes, edges)


@pytest.mark.parametrize("name", ["myciel4", "queen5_5", "queen6_6"])
def test_annealing_reaches_known_chromatic_number(name):
    G, chi = benchmarks.load(name)
    assert gc.simulated_annealing(G, chi, 300_000, seed=0).solved
    assert gc.num_colors(gc.dsatur(G)) >= chi


def test_mycielski_is_triangle_free():
    G, chi = benchmarks.load("myciel4")
    assert gc.clique_lower_bound(G) == 2 < chi
