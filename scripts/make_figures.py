"""Regenerate every figure in ``docs/figures`` and the benchmark table.

Usage::

    python scripts/make_figures.py            # everything (a few minutes)
    python scripts/make_figures.py maps gif   # only some figures
"""

from __future__ import annotations

import io
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import graph_coloring as gc  # noqa: E402
from graph_coloring import benchmarks  # noqa: E402
from graph_coloring.apps import frequency, maps, sudoku, timetabling  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET = maps.PALETTE[:4]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def save(fig, name):
    path = FIG / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# The original (2022) algorithm, reproduced faithfully for comparison: the
# acceptance exponent has the wrong sign (worse moves are always accepted),
# the temperature only cools on improvement, and H is recomputed from scratch.
# ---------------------------------------------------------------------------


def legacy_annealing(G, C, T_inicial, b, nf, seed=0):
    rng = np.random.default_rng(seed)
    H = gc.conflicts
    coloreo_actual = {v: int(rng.integers(1, C + 1)) for v in G}
    costo_actual = H(G, coloreo_actual)
    costo_nuevo = costo_actual
    T = T_inicial
    costos = np.zeros(nf)
    u = rng.uniform(0, 1, nf)
    nodes = list(coloreo_actual)
    for i in range(nf):
        cambio_nodo = nodes[rng.integers(len(nodes))]
        color_vecino = coloreo_actual.copy()
        color_vecino[cambio_nodo] = int(rng.integers(1, C + 1))
        costo_vecino = H(G, color_vecino)
        with np.errstate(over="ignore"):
            accept = u[i] < np.exp(-(costo_actual - costo_vecino) * T ** (-1))
        if costo_vecino < costo_actual or accept:
            coloreo_actual = color_vecino
            costo_actual = costo_vecino
        if costo_actual < costo_nuevo:
            costo_nuevo = costo_actual
            T *= b
        costos[i] = costo_nuevo
    return costos


def fig_before_after():
    print("before/after")
    G = nx.erdos_renyi_graph(100, 0.2, seed=1)
    k, n = 8, 30_000
    t = time.time()
    legacy = legacy_annealing(G, k, 3, 0.1, n)
    t_legacy = time.time() - t
    t = time.time()
    rnd = gc.simulated_annealing(G, k, n, move="random", seed=0, stop_when_solved=False)
    t_rnd = time.time() - t
    t = time.time()
    con = gc.simulated_annealing(G, k, n, seed=0, stop_when_solved=False)
    t_con = time.time() - t
    print(
        f"    time for {n} iters: legacy {t_legacy:.1f}s, "
        f"random {t_rnd:.2f}s, conflict {t_con:.2f}s"
    )

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(1, n + 1)
    ax.plot(x, legacy, color=ORANGE, label="Original (2022)")
    ax.plot(x, rnd.best_history[:n], color=VIOLET, label="Fixed, random node moves")
    ax.plot(x, con.best_history[:n], color=BLUE, label="Fixed, conflict-directed moves")
    ax.set_xscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best H(x) so far")
    ax.set_title(f"Coloring G(100, 0.2) with k = {k} colors")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left")
    save(fig, "before_after.png")
    return {"legacy": t_legacy, "random": t_rnd, "conflict": t_con, "n": n}


def fig_trace():
    print("trace")
    G = nx.erdos_renyi_graph(100, 0.2, seed=1)
    n = 500_000
    res = gc.simulated_annealing(
        G, 7, n, gc.geometric_for(n, 1.0, 0.1), seed=3, record_every=250, stop_when_solved=False
    )
    x = np.arange(len(res.cost_history)) * res.record_every
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True, height_ratios=[3, 1])
    a1.plot(x, res.cost_history, color=BLUE, linewidth=0.8, alpha=0.6, label="Current state")
    a1.plot(x, res.best_history, color=INK, linewidth=2, label="Best so far")
    a1.set_ylabel("H(x)")
    a1.set_title(
        f"Annealing on G(100, 0.2), k = 7: H = 0 first reached at iteration {res.solved_at:,}"
    )
    a1.axvline(res.solved_at, color=MUTED, linestyle="--", linewidth=1)
    a1.legend(loc="upper right")
    a1.set_ylim(bottom=0)
    a2.plot(x, res.temperature_history, color=MUTED)
    a2.set_ylabel("T")
    a2.set_xlabel("Iteration")
    a2.set_ylim(bottom=0)
    save(fig, "annealing_trace.png")


def fig_phase_transition():
    print("phase transition")
    cs = np.arange(3.0, 6.01, 0.25)
    budget, trials = 200_000, 20
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for n, color in [(100, ORANGE), (300, BLUE)]:
        frac = []
        for c in cs:
            ok = 0
            for t in range(trials):
                G = nx.fast_gnp_random_graph(n, c / (n - 1), seed=int(1000 * c) + t)
                ok += gc.simulated_annealing(G, 3, budget, seed=t).solved
            frac.append(ok / trials)
            print(f"    n={n} c={c:.2f} solved {ok}/{trials}")
        ax.plot(cs, frac, color=color, marker="o", markersize=5, label=f"n = {n}")
    ax.axvline(4.69, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate("c ≈ 4.69", (4.72, 0.9), color=MUTED)
    ax.set_xlabel("Average degree c")
    ax.set_ylabel("Fraction 3-colored")
    ax.set_title("3-colorability of G(n, c/n): a phase transition")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower left")
    save(fig, "phase_transition.png")


def fig_maps():
    print("maps")
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5), width_ratios=[1.1, 0.8])
    for ax, name in zip(axes, ("south_america", "chile_regions"), strict=True):
        m = maps.load_map(name)
        s = gc.estimate_chromatic_number(m.graph, seed=0)
        is_chile = name == "chile_regions"
        labels = maps.CHILE_SHORT_NAMES if is_chile else True
        maps.plot_map(
            m,
            s.coloring,
            ax=ax,
            labels=labels,
            title=f"{m.title}: {s.upper} colors",
            min_label_area=0.0 if is_chile else 12.0,
            callouts=is_chile,
        )
    save(fig, "maps.png")


GREATER_SANTIAGO = (-70.86, -70.47, -33.66, -33.31)


def fig_santiago():
    print("santiago")
    from matplotlib.patches import Rectangle

    m = maps.load_map("santiago_communes")
    s = gc.estimate_chromatic_number(m.graph, seed=0)
    x0, x1, y0, y1 = GREATER_SANTIAGO
    inside = {
        r
        for r, g in m.geometries.items()
        if x0 <= maps._label_point(g)[0] <= x1 and y0 <= maps._label_point(g)[1] <= y1
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.5), width_ratios=[1, 1.15])
    maps.plot_map(
        m,
        s.coloring,
        ax=a1,
        labels={r: "" for r in inside},
        min_label_area=0.012,
        title=f"Santiago Metropolitan Region: {len(m.graph)} communes, {s.upper} colors",
    )
    a1.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=INK, lw=1.2, ls="--", zorder=3)
    )
    maps.plot_map(
        m,
        s.coloring,
        ax=a2,
        labels=maps.SANTIAGO_SHORT_NAMES,
        extent=GREATER_SANTIAGO,
        fontsize=6.5,
        title="Greater Santiago",
        label_offsets={
            "San Joaquín": (0.012, 0.0),
            "San Miguel": (-0.006, 0.0),
            "La Granja": (0.008, 0.0),
            "San Ramón": (-0.004, -0.004),
        },
    )
    save(fig, "santiago.png")
    print(f"    chi = {s.upper}, proven: {s.exact}")


def fig_map_gif():
    print("map gif")
    m = maps.load_map("santiago_communes")
    G = m.graph
    rng = random.Random(1)
    coloring = gc.random_coloring(G, 4, rng)
    n_frames, moves_per_frame, T0, T1 = 70, 5, 0.8, 0.05
    frames = []
    for i in range(n_frames):
        cost = gc.conflicts(G, coloring)
        T = T0 * (T1 / T0) ** (i / (n_frames - 1))
        fig, ax = plt.subplots(figsize=(6, 5))
        maps.plot_map(m, coloring, ax=ax, title=f"4 colors · T = {T:.2f} · H(x) = {cost}")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor="white")
        plt.close(fig)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
        if cost == 0:
            frames += [frames[-1]] * 12  # linger on the solution
            break
        res = gc.simulated_annealing(
            G,
            4,
            moves_per_frame,
            gc.Constant(T),
            initial=coloring,
            seed=rng.randrange(1 << 30),
            stop_when_solved=False,
        )
        coloring = res.final_coloring  # the chain's state, not the best one
    path = FIG / "santiago_annealing.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=300, loop=0)
    print(f"  wrote {path.relative_to(ROOT)} ({len(frames)} frames)")


def _draw_grid(ax, grid, clues, title):
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.plot([i, i], [0, 9], color=INK, linewidth=lw)
        ax.plot([0, 9], [i, i], color=INK, linewidth=lw)
    for r in range(9):
        for c in range(9):
            if grid[r, c]:
                is_clue = clues[r, c] > 0
                ax.text(
                    c + 0.5,
                    r + 0.54,
                    str(grid[r, c]),
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold" if is_clue else "normal",
                    color=INK if is_clue else BLUE,
                )
    ax.set_title(title)


def fig_sudoku():
    print("sudoku")
    puzzle = sudoku.parse(sudoku.PUZZLES["hard"])
    t = time.time()
    res = sudoku.solve(puzzle, seed=0)
    dt = time.time() - t
    print(
        f"    solved={res.solved} in {res.iterations:,} iterations, {res.restarts} runs, {dt:.1f}s"
    )
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.6))
    _draw_grid(a1, puzzle, puzzle, "Arto Inkala's “world's hardest Sudoku”")
    _draw_grid(a2, res.grid, puzzle, f"Solved by annealing ({res.iterations / 1e6:.1f}M moves)")
    save(fig, "sudoku.png")
    return {"iterations": res.iterations, "restarts": res.restarts, "seconds": dt}


def fig_frequency():
    print("frequency")
    r = 0.1
    G = frequency.transmitters(150, r, seed=0)
    s = gc.estimate_chromatic_number(G, seed=0)
    greedy_k = gc.num_colors(gc.greedy(G))
    fig, ax = plt.subplots(figsize=(6, 6))
    frequency.plot_channels(
        G,
        s.coloring,
        ax=ax,
        radius=r,
        title=(
            f"{G.number_of_nodes()} transmitters, {s.upper} channels "
            f"(clique bound {s.lower}, greedy {greedy_k})"
        ),
    )
    save(fig, "frequency.png")


def fig_timetable():
    print("timetable")
    f = timetabling.random_faculty(seed=0)
    G = timetabling.conflict_graph(f)
    s = gc.estimate_chromatic_number(G, seed=0)
    table = timetabling.timetable(s.coloring)
    print(f"    {len(f.courses)} courses, {f.n_students} students, {s.upper} slots (≥ {s.lower})")
    prog_color = dict(zip(("MAT", "FIS", "AST", "INF", "EST"), maps.PALETTE[:5], strict=True))
    days, blocks = timetabling.DAYS, 6
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, len(days))
    ax.set_ylim(blocks, 0)
    ax.grid(False)
    ax.set_xticks(np.arange(len(days)) + 0.5, days)
    ax.set_yticks(np.arange(blocks) + 0.5, [f"Block {b + 1}" for b in range(blocks)])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for d in range(len(days) + 1):
        ax.axvline(d, color=GRID, linewidth=1)
    for b in range(blocks + 1):
        ax.axhline(b, color=GRID, linewidth=1)
    for slot_name, courses in table.items():
        day, block = slot_name.split("-")
        x, y = days.index(day), int(block) - 1
        for j, course in enumerate(courses):
            ax.text(
                x + 0.06 + (j % 3) * 0.31,
                y + 0.3 + (j // 3) * 0.3,
                course,
                fontsize=8,
                color="white",
                va="center",
                bbox={"boxstyle": "round,pad=0.25", "fc": prog_color[course[:3]], "ec": "none"},
            )
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=c)
        for c in prog_color.values()
    ]
    ax.legend(handles, list(prog_color), loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=5)
    ax.set_title(
        f"Clash-free weekly timetable: {len(f.courses)} courses, {f.n_students} students, "
        f"{s.upper} time slots"
    )
    save(fig, "timetable.png")


def benchmark_table():
    print("benchmarks")
    rows = [
        "| Graph | n | m | ω (clique) | Greedy | DSATUR | Annealing | Proof of χ | Known χ |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for name in benchmarks.KNOWN:
        G, chi = benchmarks.load(name)
        s = gc.estimate_chromatic_number(G, n_iter=500_000, restarts=3, seed=0)
        omega = gc.clique_lower_bound(G)
        how = "clique" if omega == s.upper else "exact search" if s.exact else "no"
        rows.append(
            f"| `{name}` | {G.number_of_nodes()} | {G.number_of_edges()} | {omega} | "
            f"{gc.num_colors(gc.greedy(G))} | {gc.num_colors(gc.dsatur(G))} | "
            f"**{s.upper}** | {how} | {chi} |"
        )
    text = "\n".join(rows) + "\n"
    (ROOT / "docs" / "benchmarks.md").write_text(text, encoding="utf-8")
    print(text)


FIGURES = {
    "before_after": fig_before_after,
    "trace": fig_trace,
    "phase": fig_phase_transition,
    "maps": fig_maps,
    "santiago": fig_santiago,
    "gif": fig_map_gif,
    "sudoku": fig_sudoku,
    "frequency": fig_frequency,
    "timetable": fig_timetable,
    "benchmarks": benchmark_table,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGURES)
    for key in wanted:
        out = FIGURES[key]()
        if out:
            print("   ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()})
