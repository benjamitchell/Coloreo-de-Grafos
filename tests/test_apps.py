import networkx as nx
import numpy as np
import pytest

import graph_coloring as gc
from graph_coloring.apps import frequency, maps, sudoku, timetabling

# --- Sudoku -----------------------------------------------------------------


def test_sudoku_graph_structure():
    G = sudoku.sudoku_graph()
    assert G.number_of_nodes() == 81
    assert G.number_of_edges() == 810
    assert all(d == 20 for _, d in G.degree)
    assert gc.clique_lower_bound(G) == 9


def test_parse_and_print_roundtrip():
    grid = sudoku.parse(sudoku.PUZZLES["easy"])
    assert grid.shape == (9, 9)
    assert grid[0, 0] == 5 and grid[0, 2] == 0
    text = sudoku.to_string(grid)
    assert np.array_equal(
        sudoku.parse(text.replace("|", "").replace("-", "").replace("+", "")), grid
    )
    with pytest.raises(ValueError):
        sudoku.parse("123")


@pytest.mark.parametrize("method", ["box-swap", "recolor"])
@pytest.mark.parametrize("name", ["easy", "medium"])
def test_solve_easy_sudokus(method, name):
    res = sudoku.solve(sudoku.PUZZLES[name], method=method, use_propagation=False, seed=0)
    assert res.solved
    assert sudoku.is_valid_solution(res.grid, res.puzzle)
    assert sudoku.grid_conflicts(res.grid) == 0


def test_propagation_solves_easy_sudoku():
    grid = sudoku.propagate(sudoku.parse(sudoku.PUZZLES["easy"]))
    assert sudoku.is_valid_solution(grid)


def test_propagation_detects_contradiction():
    bad = sudoku.parse(sudoku.PUZZLES["easy"])
    bad[0, 2] = 5  # a second 5 in the first row
    with pytest.raises(ValueError):
        sudoku.propagate(bad)


def test_is_valid_solution_rejects_changed_clues():
    solution = sudoku.solve(sudoku.PUZZLES["easy"], seed=0).grid
    puzzle = sudoku.parse(sudoku.PUZZLES["easy"])
    assert sudoku.is_valid_solution(solution, puzzle)
    other = puzzle.copy()
    other[0, 0] = 1
    assert not sudoku.is_valid_solution(solution, other)


def test_box_swap_cost_matches_graph_cost():
    res = sudoku.solve(sudoku.PUZZLES["hard"], n_iter=2_000, restarts=1, seed=0)
    assert res.cost == sudoku.grid_conflicts(res.grid)


# --- Maps -------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, regions, chi", [("south_america", 13, 4), ("chile_regions", 16, 3), ("us_states", 48, 4)]
)
def test_maps_are_planar_with_expected_chromatic_number(name, regions, chi):
    m = maps.load_map(name)
    G = m.graph
    assert G.number_of_nodes() == regions
    assert nx.check_planarity(G)[0]
    assert set(m.geometries) == set(G)
    assert gc.simulated_annealing(G, chi, 100_000, seed=0).solved
    assert gc.k_coloring(G, chi - 1)[0] == "infeasible"  # chi is proven, not just observed


def test_map_adjacency_facts():
    sa = maps.load_map("south_america").graph
    assert nx.is_isomorphic(
        sa.subgraph(["Argentina", "Bolivia", "Brazil", "Paraguay"]), nx.complete_graph(4)
    )
    assert not sa.has_edge("Chile", "Ecuador")
    us = maps.load_map("us_states").graph
    assert us.has_edge("Utah", "Arizona")
    assert not us.has_edge("Utah", "New Mexico")  # Four Corners: a point, not a border
    assert not us.has_edge("Michigan", "Illinois")  # only across Lake Michigan
    stgo = maps.load_map("santiago_communes").graph
    assert stgo.has_edge("Santiago", "Providencia")
    assert not stgo.has_edge("Renca", "Santiago")  # they only meet at a corner
    wheel = ["Maipú", "Padre Hurtado", "Peñaflor", "Talagante", "San Bernardo"]
    assert set(stgo.adj["Calera de Tango"]) == set(wheel)
    assert nx.is_isomorphic(stgo.subgraph(wheel), nx.cycle_graph(5))
    cl = maps.load_map("chile_regions").graph
    assert nx.is_connected(cl)
    assert cl.degree("Arica y Parinacota") == 1


def test_plot_map_runs():
    import matplotlib

    matplotlib.use("Agg")
    m = maps.load_map("south_america")
    ax = maps.plot_map(m, gc.dsatur(m.graph), labels=True, title="t")
    assert ax.get_title() == "t"


def test_unknown_map():
    with pytest.raises(ValueError):
        maps.load_map("atlantis")


# --- Frequency assignment ---------------------------------------------------


def test_transmitters_interference_graph():
    G = frequency.transmitters(100, 0.15, seed=0)
    pos = nx.get_node_attributes(G, "pos")
    for u, v in G.edges:
        assert np.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]) <= 0.15
    Gc = frequency.transmitters(100, 0.15, clusters=3, seed=0)
    assert Gc.number_of_edges() > G.number_of_edges()


def test_channel_assignment_is_near_clique_bound():
    G = frequency.transmitters(120, 0.12, seed=1)
    s = gc.estimate_chromatic_number(G, n_iter=50_000, seed=0)
    assert gc.is_proper(G, s.coloring)
    assert s.upper <= s.lower + 1


# --- Timetabling ------------------------------------------------------------


def test_timetabling_conflict_graph():
    f = timetabling.random_faculty(seed=0)
    G = timetabling.conflict_graph(f)
    assert set(G) == set(f.courses)
    a, b, data = next(iter(G.edges(data=True)))
    shared = len(f.enrolment[a] & f.enrolment[b])
    assert data["students"] == shared
    assert data["lecturer"] == (f.lecturer[a] == f.lecturer[b])
    assert shared > 0 or data["lecturer"]


def test_timetable_from_coloring():
    f = timetabling.random_faculty(programs=("MAT", "FIS"), seed=1)
    G = timetabling.conflict_graph(f)
    coloring = gc.dsatur(G)
    assert timetabling.clashes(G, coloring) == 0
    table = timetabling.timetable(coloring)
    assert sum(len(v) for v in table.values()) == len(f.courses)
    assert next(iter(table)) == "Mon-1"


def test_slot_names():
    assert timetabling.slot_name(0) == "Mon-1"
    assert timetabling.slot_name(7) == "Tue-2"
    assert timetabling.slot_name(30) == "Extra1-1"
