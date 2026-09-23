"""Sudoku as graph coloring.

A 9x9 Sudoku is a 9-coloring problem on the *Sudoku graph*: 81 nodes
(cells), with an edge between any two cells in the same row, column or
3x3 box. Every cell has 8 + 8 + 4 = 20 neighbours, for 810 edges. The
clues are nodes whose color is fixed, and a solution is exactly a proper
9-coloring that extends them.

Two annealing neighbourhoods are provided:

* ``method="recolor"``: the generic solver from :mod:`graph_coloring.annealing`,
  which recolors one conflicting cell at a time with the clues fixed.
* ``method="box-swap"`` (Lewis, 2007): every 3x3 box starts as a
  permutation of its missing digits and a move swaps two free cells of the
  same box. Box constraints then hold by construction, only row and column
  conflicts remain, and the search space is much smaller. This is still
  graph coloring: we simply restrict the state space to colorings that are
  proper on the box cliques.

Optionally, constraint propagation (naked and hidden singles) runs first
and fixes many more cells.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np

from ..annealing import simulated_annealing
from ..schedules import Constant, Schedule

Cell = tuple[int, int]


def sudoku_graph(box: int = 3) -> nx.Graph:
    """Return the Sudoku graph for a ``box**2 x box**2`` grid.

    Nodes are ``(row, col)`` pairs. ``box=3`` gives the usual 9x9 Sudoku.
    """
    n = box * box
    G = nx.Graph()
    cells = [(r, c) for r in range(n) for c in range(n)]
    G.add_nodes_from(cells)
    for r1, c1 in cells:
        for r2, c2 in cells:
            if (r1, c1) < (r2, c2) and (
                r1 == r2 or c1 == c2 or (r1 // box == r2 // box and c1 // box == c2 // box)
            ):
                G.add_edge((r1, c1), (r2, c2))
    return G


def parse(puzzle: str | np.ndarray | list[list[int]]) -> np.ndarray:
    """Parse a puzzle into a 9x9 int array with 0 for empty cells.

    Strings may use digits and ``.`` or ``0`` for blanks; whitespace and
    other separators are ignored.
    """
    if isinstance(puzzle, str):
        chars = [ch for ch in puzzle if ch.isdigit() or ch == "."]
        grid = np.array([0 if ch == "." else int(ch) for ch in chars], dtype=int)
    else:
        grid = np.asarray(puzzle, dtype=int).ravel()
    size = int(round(len(grid) ** 0.5))
    if size * size != len(grid) or int(round(size**0.5)) ** 2 != size:
        raise ValueError(f"expected a square grid with a square side, got {len(grid)} cells")
    return grid.reshape(size, size)


def to_string(grid: np.ndarray) -> str:
    """Pretty-print a 9x9 grid with box separators."""
    n = grid.shape[0]
    box = int(round(n**0.5))
    lines = []
    for r in range(n):
        if r and r % box == 0:
            lines.append("+".join(["-" * (2 * box + 1)] * box))
        row = []
        for c in range(n):
            if c and c % box == 0:
                row.append("|")
            row.append(str(grid[r, c]) if grid[r, c] else ".")
        lines.append(" " + " ".join(row))
    return "\n".join(lines)


def is_valid_solution(grid: np.ndarray, puzzle: np.ndarray | None = None) -> bool:
    """Check that ``grid`` is a complete valid Sudoku (extending ``puzzle``)."""
    n = grid.shape[0]
    box = int(round(n**0.5))
    digits = set(range(1, n + 1))
    for i in range(n):
        if set(grid[i, :]) != digits or set(grid[:, i]) != digits:
            return False
    for br in range(0, n, box):
        for bc in range(0, n, box):
            if set(grid[br : br + box, bc : bc + box].ravel()) != digits:
                return False
    if puzzle is not None:
        mask = puzzle > 0
        if not np.array_equal(grid[mask], puzzle[mask]):
            return False
    return True


def propagate(grid: np.ndarray) -> np.ndarray:
    """Fill cells forced by naked singles and hidden singles.

    A *naked single* is a cell with only one candidate left; a *hidden
    single* is a digit that fits in only one cell of a row, column or box.
    Returns a new grid (the input is not modified). Raises ``ValueError``
    if a contradiction is found.
    """
    grid = grid.copy()
    n = grid.shape[0]
    box = int(round(n**0.5))
    G = sudoku_graph(box)
    units = (
        [[(r, c) for c in range(n)] for r in range(n)]
        + [[(r, c) for r in range(n)] for c in range(n)]
        + [
            [(br + i, bc + j) for i in range(box) for j in range(box)]
            for br in range(0, n, box)
            for bc in range(0, n, box)
        ]
    )

    def candidates(cell: Cell) -> set[int]:
        return set(range(1, n + 1)) - {grid[nb] for nb in G.adj[cell]}

    changed = True
    while changed:
        changed = False
        for cell in G.nodes:
            if grid[cell] == 0:
                cand = candidates(cell)
                if not cand:
                    raise ValueError(f"no candidates left for cell {cell}")
                if len(cand) == 1:
                    grid[cell] = cand.pop()
                    changed = True
        for unit in units:
            for d in range(1, n + 1):
                if any(grid[cell] == d for cell in unit):
                    continue
                spots = [cell for cell in unit if grid[cell] == 0 and d in candidates(cell)]
                if len(spots) == 1:
                    grid[spots[0]] = d
                    changed = True
                elif not spots:
                    raise ValueError(f"digit {d} has no place in a unit")
    return grid


@dataclass
class SudokuResult:
    puzzle: np.ndarray
    grid: np.ndarray
    """Best grid found (a valid solution when ``solved``)."""
    solved: bool
    propagated: int
    """Number of cells filled by constraint propagation before annealing."""
    cost: int
    """Conflicting edges ``H`` of ``grid`` in the Sudoku graph."""
    iterations: int
    """Total annealing iterations over all restarts."""
    restarts: int
    """Number of annealing runs performed."""
    cost_history: np.ndarray
    """Cost of the current state for the last run, sampled regularly."""


def grid_conflicts(grid: np.ndarray) -> int:
    """``H`` of a complete grid: pairs of equal digits sharing a unit."""
    box = int(round(grid.shape[0] ** 0.5))
    G = sudoku_graph(box)
    return sum(1 for a, b in G.edges if grid[a] == grid[b])


def solve(
    puzzle: str | np.ndarray | list[list[int]],
    method: Literal["box-swap", "recolor"] = "box-swap",
    n_iter: int = 1_000_000,
    restarts: int = 20,
    schedule: Schedule | None = None,
    use_propagation: bool = True,
    seed: int | None = None,
    record_every: int = 100,
) -> SudokuResult:
    """Solve a Sudoku by simulated annealing on the Sudoku graph.

    Each restart runs up to ``n_iter`` iterations; the best grid over all
    restarts is returned.
    """
    puzzle = parse(puzzle)
    start = propagate(puzzle) if use_propagation else puzzle
    propagated = int((start > 0).sum() - (puzzle > 0).sum())
    rng = random.Random(seed)

    best_grid, best_cost = None, math.inf
    total_iter, history = 0, np.zeros(0, dtype=np.int64)
    runs = 0
    for _ in range(restarts):
        runs += 1
        if method == "box-swap":
            grid, cost, it, history = _anneal_box_swap(
                start, n_iter, schedule or Constant(0.5), rng, record_every
            )
        elif method == "recolor":
            grid, cost, it, history = _anneal_recolor(
                start, n_iter, schedule, rng.randrange(2**32), record_every
            )
        else:
            raise ValueError(f"unknown method {method!r}")
        total_iter += it
        if cost < best_cost:
            best_grid, best_cost = grid, cost
        if cost == 0:
            break

    return SudokuResult(
        puzzle=puzzle,
        grid=best_grid,
        solved=best_cost == 0,
        propagated=propagated,
        cost=int(best_cost),
        iterations=total_iter,
        restarts=runs,
        cost_history=history,
    )


def _anneal_recolor(start, n_iter, schedule, seed, record_every):
    n = start.shape[0]
    G = sudoku_graph(int(round(n**0.5)))
    fixed = {cell: int(start[cell]) - 1 for cell in G.nodes if start[cell]}
    res = simulated_annealing(
        G, n, n_iter, schedule, fixed=fixed, seed=seed, record_every=record_every
    )
    grid = np.zeros_like(start)
    for cell, color in res.coloring.items():
        grid[cell] = color + 1
    return grid, res.cost, res.iterations, res.cost_history


def _anneal_box_swap(start, n_iter, schedule, rng, record_every):
    n = start.shape[0]
    box = int(round(n**0.5))
    grid = start.copy()
    G = sudoku_graph(box)
    # Digits allowed in each free cell: those not clashing with a fixed cell.
    cand = {
        cell: set(range(1, n + 1)) - {int(start[nb]) for nb in G.adj[cell]}
        for cell in G.nodes
        if start[cell] == 0
    }
    free_by_box: list[list[Cell]] = []
    for br in range(0, n, box):
        for bc in range(0, n, box):
            cells = [(br + i, bc + j) for i in range(box) for j in range(box)]
            free = [cell for cell in cells if grid[cell] == 0]
            missing = set(range(1, n + 1)) - {int(grid[cell]) for cell in cells}
            for cell, d in _random_matching(free, missing, cand, rng).items():
                grid[cell] = d
            if len(free) >= 2:
                free_by_box.append(free)

    # row_cnt[r][d] = occurrences of digit d in row r (same for columns);
    # H = sum over rows and columns of C(count, 2).
    row_cnt = [[0] * (n + 1) for _ in range(n)]
    col_cnt = [[0] * (n + 1) for _ in range(n)]
    for r in range(n):
        for c in range(n):
            row_cnt[r][grid[r, c]] += 1
            col_cnt[c][grid[r, c]] += 1
    cost = sum(x * (x - 1) // 2 for cnt in row_cnt + col_cnt for x in cnt)

    history = np.empty(n_iter // record_every + 1, dtype=np.int64)
    rec = 0
    it = 0
    if not free_by_box or cost == 0:
        history[0] = cost
        return grid, cost, 0, history[:1]

    for it in range(n_iter):
        if it % record_every == 0:
            history[rec] = cost
            rec += 1
        free = free_by_box[int(rng.random() * len(free_by_box))]
        a, b = rng.sample(free, 2)
        (r1, c1), (r2, c2) = a, b
        d1, d2 = int(grid[a]), int(grid[b])
        if d2 not in cand[a] or d1 not in cand[b]:
            continue

        # Remove d1 from (r1, c1) and d2 from (r2, c2), then add them swapped.
        delta = 0
        if r1 != r2:
            delta += (row_cnt[r1][d2] - (row_cnt[r1][d1] - 1)) + (
                row_cnt[r2][d1] - (row_cnt[r2][d2] - 1)
            )
        if c1 != c2:
            delta += (col_cnt[c1][d2] - (col_cnt[c1][d1] - 1)) + (
                col_cnt[c2][d1] - (col_cnt[c2][d2] - 1)
            )

        if delta <= 0 or rng.random() < math.exp(-delta / schedule(it)):
            grid[a], grid[b] = d2, d1
            if r1 != r2:
                row_cnt[r1][d1] -= 1
                row_cnt[r1][d2] += 1
                row_cnt[r2][d2] -= 1
                row_cnt[r2][d1] += 1
            if c1 != c2:
                col_cnt[c1][d1] -= 1
                col_cnt[c1][d2] += 1
                col_cnt[c2][d2] -= 1
                col_cnt[c2][d1] += 1
            cost += delta
            if cost == 0:
                break

    history[rec] = cost
    return grid, cost, it + 1, history[: rec + 1]


def _random_matching(
    cells: list[Cell], digits: set[int], cand: dict[Cell, set[int]], rng: random.Random
) -> dict[Cell, int]:
    """Assign ``digits`` to ``cells`` bijectively, respecting ``cand`` if
    possible, choosing among perfect matchings at random."""
    cells = cells[:]
    rng.shuffle(cells)
    B = nx.Graph()
    B.add_nodes_from(cells)
    B.add_nodes_from(("d", d) for d in digits)
    edges = [(cell, ("d", d)) for cell in cells for d in cand[cell] if d in digits]
    rng.shuffle(edges)
    B.add_edges_from(edges)
    matching = nx.bipartite.hopcroft_karp_matching(B, top_nodes=cells)
    if all(cell in matching for cell in cells):
        return {cell: matching[cell][1] for cell in cells}
    # No candidate-respecting fill exists (contradictory puzzle): ignore cand.
    rest = list(digits)
    rng.shuffle(rest)
    return dict(zip(cells, rest, strict=True))


# A few well-known puzzles, from easy to very hard.
PUZZLES = {
    "easy": ("53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"),
    "medium": ("...26.7.168..7..9.19...45..82.1...4...46.29...5...3.28..93...74.4..5..367.3.18..."),
    # Arto Inkala's "world's hardest Sudoku" (2012)
    "hard": ("8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4.."),
}
