"""Simulated annealing for the k-coloring problem.

We minimise the number of conflicting edges ``H(x)`` over colorings
``x: V -> {0, ..., k-1}`` using the Metropolis acceptance rule

    P(accept x -> x') = min(1, exp(-(H(x') - H(x)) / T))

with a temperature ``T`` that decreases along a cooling schedule.

A move recolors a single node. Since only the edges incident to that node
change, ``H(x') - H(x)`` is computed in ``O(deg(v))`` time, and the set of
nodes currently in conflict is maintained incrementally so that moves can
be restricted to them (``move="conflict"``), which is much more effective
than picking nodes uniformly at random.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np

from .schedules import Schedule, geometric_for


@dataclass
class AnnealingResult:
    """Outcome of a simulated annealing run."""

    coloring: dict[Hashable, int] = field(repr=False)
    """Best coloring found (colors are ``0..k-1``)."""
    cost: int
    """Number of conflicting edges of ``coloring``."""
    k: int
    """Number of colors available."""
    iterations: int
    """Number of iterations actually performed."""
    solved_at: int | None
    """Iteration where a proper coloring was first reached, if ever."""
    cost_history: np.ndarray = field(repr=False)
    """Cost of the current state, sampled every ``record_every`` iterations."""
    best_history: np.ndarray = field(repr=False)
    """Best cost so far, sampled every ``record_every`` iterations."""
    temperature_history: np.ndarray = field(repr=False)
    """Temperature, sampled every ``record_every`` iterations."""
    record_every: int = 1
    final_coloring: dict[Hashable, int] = field(default_factory=dict, repr=False)
    """State of the Markov chain after the last iteration (may be worse than
    ``coloring``). Useful to continue a run or to animate it."""

    @property
    def solved(self) -> bool:
        return self.cost == 0


class _IndexedSet:
    """Set of ints with O(1) insert, remove and uniform random choice."""

    __slots__ = ("items", "pos")

    def __init__(self, n: int) -> None:
        self.items: list[int] = []
        self.pos = [-1] * n

    def __len__(self) -> int:
        return len(self.items)

    def add(self, x: int) -> None:
        if self.pos[x] < 0:
            self.pos[x] = len(self.items)
            self.items.append(x)

    def discard(self, x: int) -> None:
        i = self.pos[x]
        if i >= 0:
            last = self.items.pop()
            if last != x:
                self.items[i] = last
                self.pos[last] = i
            self.pos[x] = -1

    def choice(self, rng: random.Random) -> int:
        return self.items[int(rng.random() * len(self.items))]


def random_coloring(
    G: nx.Graph, k: int, seed: int | random.Random | None = None
) -> dict[Hashable, int]:
    """Assign each node a uniformly random color in ``0..k-1``."""
    rng = seed if isinstance(seed, random.Random) else random.Random(seed)
    return {v: rng.randrange(k) for v in G.nodes}


def simulated_annealing(
    G: nx.Graph,
    k: int,
    n_iter: int = 100_000,
    schedule: Schedule | None = None,
    *,
    initial: Mapping[Hashable, int] | None = None,
    fixed: Mapping[Hashable, int] | None = None,
    move: Literal["conflict", "random"] = "conflict",
    seed: int | None = None,
    stop_when_solved: bool = True,
    record_every: int = 1,
) -> AnnealingResult:
    """Search for a proper ``k``-coloring of ``G`` by simulated annealing.

    Parameters
    ----------
    G:
        Undirected simple graph.
    k:
        Number of available colors, labelled ``0..k-1``.
    n_iter:
        Maximum number of iterations (proposed moves).
    schedule:
        Cooling schedule ``T(i)``. Defaults to a geometric schedule from
        ``T=0.6`` to ``T=0.1`` over ``n_iter`` iterations, which works well
        with ``move="conflict"`` on a wide range of graphs.
    initial:
        Starting coloring. Random if omitted.
    fixed:
        Nodes whose color is fixed and never changed (e.g. Sudoku clues).
        Overrides ``initial`` on those nodes.
    move:
        ``"conflict"`` recolors a random node that is currently in conflict
        (min-conflicts style neighbourhood); ``"random"`` recolors a
        uniformly random node, as in the textbook algorithm.
    seed:
        Seed for reproducibility.
    stop_when_solved:
        Stop as soon as a proper coloring is found.
    record_every:
        Store the cost/temperature history every this many iterations.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if schedule is None:
        schedule = geometric_for(n_iter)
    rng = random.Random(seed)
    fixed = dict(fixed or {})

    nodes = list(G.nodes)
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    adj = [[index[w] for w in G.adj[v] if w != v] for v in nodes]

    start = dict(initial) if initial is not None else random_coloring(G, k, rng)
    start.update(fixed)
    col = [start[v] for v in nodes]
    if any(not 0 <= c < k for c in col):
        raise ValueError(f"colors must lie in 0..{k - 1}")
    movable = [v not in fixed for v in nodes]
    movable_nodes = [i for i in range(n) if movable[i]]

    # conf[v] = number of neighbours of v sharing its color
    conf = [sum(1 for w in adj[v] if col[w] == col[v]) for v in range(n)]
    cost = sum(conf) // 2
    in_conflict = _IndexedSet(n)
    for v in movable_nodes:
        if conf[v]:
            in_conflict.add(v)

    best_cost, best_col = cost, col[:]
    solved_at = 0 if cost == 0 else None

    n_rec = n_iter // record_every + 1
    cost_hist = np.empty(n_rec, dtype=np.int64)
    best_hist = np.empty(n_rec, dtype=np.int64)
    temp_hist = np.empty(n_rec, dtype=np.float64)
    rec = 0

    it = 0
    if not movable_nodes or k == 1 or (stop_when_solved and cost == 0):
        n_iter_eff = 0
    else:
        n_iter_eff = n_iter

    for it in range(n_iter_eff):
        T = schedule(it)
        if it % record_every == 0:
            cost_hist[rec], best_hist[rec], temp_hist[rec] = cost, best_cost, T
            rec += 1

        if move == "conflict" and len(in_conflict):
            v = in_conflict.choice(rng)
        else:
            v = movable_nodes[int(rng.random() * len(movable_nodes))]
        old = col[v]
        new = int(rng.random() * (k - 1))
        if new >= old:
            new += 1

        delta = 0
        for w in adj[v]:
            cw = col[w]
            if cw == new:
                delta += 1
            elif cw == old:
                delta -= 1

        if delta <= 0 or rng.random() < math.exp(-delta / T):
            col[v] = new
            conf[v] = 0
            for w in adj[v]:
                cw = col[w]
                if cw == old:
                    conf[w] -= 1
                    if conf[w] == 0:
                        in_conflict.discard(w)
                elif cw == new:
                    conf[w] += 1
                    conf[v] += 1
                    if movable[w]:
                        in_conflict.add(w)
            if conf[v]:
                in_conflict.add(v)
            else:
                in_conflict.discard(v)
            cost += delta

            if cost < best_cost:
                best_cost, best_col = cost, col[:]
                if cost == 0:
                    solved_at = it + 1
                    if stop_when_solved:
                        break

    iterations = it + 1 if n_iter_eff else 0
    # final sample so that histories always end at the returned state
    cost_hist[rec], best_hist[rec], temp_hist[rec] = cost, best_cost, schedule(iterations)
    rec += 1

    return AnnealingResult(
        coloring={v: best_col[i] for i, v in enumerate(nodes)},
        cost=best_cost,
        k=k,
        iterations=iterations,
        solved_at=solved_at,
        cost_history=cost_hist[:rec],
        best_history=best_hist[:rec],
        temperature_history=temp_hist[:rec],
        record_every=record_every,
        final_coloring={v: col[i] for i, v in enumerate(nodes)},
    )
