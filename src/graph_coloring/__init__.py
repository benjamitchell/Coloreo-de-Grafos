"""Graph coloring by simulated annealing, with classical baselines and
applications (Sudoku, maps, frequency assignment, timetabling)."""

from .annealing import AnnealingResult, random_coloring, simulated_annealing
from .chromatic import (
    ChromaticSearch,
    brooks_bound,
    clique_lower_bound,
    degeneracy_bound,
    estimate_chromatic_number,
    max_degree_bound,
)
from .classical import HEURISTICS, dsatur, greedy, smallest_last, welsh_powell
from .cost import conflicting_edges, conflicts, delta_recolor, is_proper, num_colors
from .schedules import Constant, Geometric, Linear, Logarithmic, geometric_for

__all__ = [
    "AnnealingResult",
    "ChromaticSearch",
    "Constant",
    "Geometric",
    "HEURISTICS",
    "Linear",
    "Logarithmic",
    "brooks_bound",
    "clique_lower_bound",
    "conflicting_edges",
    "conflicts",
    "degeneracy_bound",
    "delta_recolor",
    "dsatur",
    "estimate_chromatic_number",
    "geometric_for",
    "greedy",
    "is_proper",
    "max_degree_bound",
    "num_colors",
    "random_coloring",
    "simulated_annealing",
    "smallest_last",
    "welsh_powell",
]

__version__ = "1.0.0"
