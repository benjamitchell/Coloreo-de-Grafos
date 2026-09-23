import networkx as nx
import numpy as np
import pytest

import graph_coloring as gc


@pytest.fixture
def G():
    return nx.erdos_renyi_graph(60, 0.15, seed=3)


def test_conflicts_and_delta_agree(G):
    rng = np.random.default_rng(0)
    coloring = {v: int(rng.integers(4)) for v in G}
    for _ in range(200):
        v = int(rng.integers(len(G)))
        c = int(rng.integers(4))
        before = gc.conflicts(G, coloring)
        d = gc.delta_recolor(G, coloring, v, c)
        coloring[v] = c
        assert gc.conflicts(G, coloring) == before + d


def test_is_proper_matches_conflicts(G):
    coloring = gc.dsatur(G)
    assert gc.is_proper(G, coloring)
    assert gc.conflicts(G, coloring) == 0
    u, v = next(iter(G.edges))
    coloring[u] = coloring[v]
    assert not gc.is_proper(G, coloring)
    assert (u, v) in gc.conflicting_edges(G, coloring) or (v, u) in gc.conflicting_edges(
        G, coloring
    )


@pytest.mark.parametrize("move", ["conflict", "random"])
def test_annealing_reported_cost_is_true_cost(G, move):
    res = gc.simulated_annealing(G, 3, 20_000, seed=1, move=move, stop_when_solved=False)
    assert res.cost == gc.conflicts(G, res.coloring)
    assert res.best_history[-1] == res.cost
    assert np.all(np.diff(res.best_history) <= 0)
    assert set(res.coloring.values()) <= set(range(3))


def test_annealing_solves_easy_instances():
    assert gc.simulated_annealing(nx.cycle_graph(20), 2, 10_000, seed=0).solved
    assert gc.simulated_annealing(nx.petersen_graph(), 3, 10_000, seed=0).solved
    res = gc.simulated_annealing(nx.complete_graph(6), 6, 10_000, seed=0)
    assert res.solved and res.solved_at is not None


def test_annealing_cannot_beat_chromatic_number():
    res = gc.simulated_annealing(nx.cycle_graph(21), 2, 20_000, seed=0)
    assert not res.solved
    assert res.cost >= 1
    assert res.solved_at is None


def test_annealing_respects_fixed_nodes():
    G = nx.path_graph(10)
    fixed = {0: 1, 5: 0, 9: 0}  # consistent with the parity of a path
    res = gc.simulated_annealing(G, 2, 10_000, fixed=fixed, seed=0)
    assert res.solved
    for v, c in fixed.items():
        assert res.coloring[v] == c


def test_annealing_is_reproducible(G):
    a = gc.simulated_annealing(G, 4, 5_000, seed=42)
    b = gc.simulated_annealing(G, 4, 5_000, seed=42)
    assert a.coloring == b.coloring
    assert np.array_equal(a.cost_history, b.cost_history)


def test_annealing_record_every(G):
    res = gc.simulated_annealing(G, 3, 10_000, seed=0, record_every=100, stop_when_solved=False)
    assert len(res.cost_history) == 10_000 // 100 + 1


def test_annealing_rejects_bad_input(G):
    with pytest.raises(ValueError):
        gc.simulated_annealing(G, 0)
    with pytest.raises(ValueError):
        gc.simulated_annealing(G, 2, initial={v: 5 for v in G})


def test_annealing_trivial_cases():
    assert gc.simulated_annealing(nx.empty_graph(5), 1, 100).solved
    assert not gc.simulated_annealing(nx.path_graph(3), 1, 100).solved
    assert gc.simulated_annealing(nx.Graph(), 3, 100).iterations == 0


@pytest.mark.parametrize(
    "schedule",
    [
        gc.Constant(0.5),
        gc.Geometric(),
        gc.Linear(n_iter=100),
        gc.Logarithmic(),
        gc.geometric_for(100),
    ],
)
def test_schedules_are_positive_and_non_increasing(schedule):
    temps = [schedule(i) for i in range(200)]
    assert all(t > 0 for t in temps)
    assert all(a >= b for a, b in zip(temps, temps[1:], strict=False))


def test_geometric_for_hits_endpoints():
    s = gc.geometric_for(1000, 2.0, 0.1)
    assert s(0) == pytest.approx(2.0)
    assert s(999) == pytest.approx(0.1)


def test_final_coloring_is_chain_state(G):
    res = gc.simulated_annealing(G, 3, 5_000, seed=0, stop_when_solved=False)
    assert res.cost_history[-1] == gc.conflicts(G, res.final_coloring)
    assert res.cost <= gc.conflicts(G, res.final_coloring)
