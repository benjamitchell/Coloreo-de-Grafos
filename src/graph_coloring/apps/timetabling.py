"""Timetabling as graph coloring.

Each course is a node and two courses are joined when they cannot happen
at the same time: they share a student or a lecturer. A proper coloring
with ``k`` colors is a clash-free timetable with ``k`` time slots, so the
fewest slots needed is the chromatic number of the conflict graph.

This is the classical formulation of exam timetabling (Welsh & Powell,
1967; Carter, Laporte & Lee, 1996) and also covers a weekly class
schedule where each course meets in one fixed slot.

The instances here are synthetic but structured like a real faculty:
students belong to a program and a year, take mostly the courses of their
own program and year, and occasionally an elective from another program.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")


@dataclass
class Faculty:
    """A synthetic faculty: courses, who takes them and who teaches them."""

    courses: list[str]
    enrolment: dict[str, set[int]]
    """Students enrolled in each course."""
    lecturer: dict[str, str]
    """Lecturer of each course."""

    @property
    def n_students(self) -> int:
        return len(set().union(*self.enrolment.values()))


def random_faculty(
    programs: tuple[str, ...] = ("MAT", "FIS", "AST", "INF", "EST"),
    years: int = 4,
    courses_per_year: int = 5,
    students_per_cohort: int = 40,
    electives: int = 2,
    lecturers_per_program: int = 8,
    seed: int | None = None,
) -> Faculty:
    """Generate a faculty. Course codes look like ``MAT201`` (program MAT,
    year 2, course 01)."""
    rng = np.random.default_rng(seed)
    courses: list[str] = []
    by_cohort: dict[tuple[str, int], list[str]] = {}
    lecturer: dict[str, str] = {}
    for prog in programs:
        staff = [f"{prog}-L{i + 1}" for i in range(lecturers_per_program)]
        for y in range(1, years + 1):
            cohort = [f"{prog}{y}{c + 1:02d}" for c in range(courses_per_year)]
            by_cohort[(prog, y)] = cohort
            courses += cohort
            for course in cohort:
                lecturer[course] = str(rng.choice(staff))

    enrolment: dict[str, set[int]] = {c: set() for c in courses}
    student = 0
    for (_, y), cohort in by_cohort.items():
        for _ in range(students_per_cohort):
            # most of the cohort's courses (some students are behind/ahead)
            take = [c for c in cohort if rng.random() < 0.85]
            # electives from other programs, same or next year
            pool = [
                c
                for (_, y2), other in by_cohort.items()
                if other is not cohort and y2 in (y, y + 1)
                for c in other
            ]
            take += list(rng.choice(pool, size=min(electives, len(pool)), replace=False))
            for c in take:
                enrolment[c].add(student)
            student += 1
    return Faculty(courses=courses, enrolment=enrolment, lecturer=lecturer)


def conflict_graph(faculty: Faculty) -> nx.Graph:
    """Courses joined when they share students (edge attribute ``students``
    counts them) or a lecturer (``lecturer=True``)."""
    G = nx.Graph()
    G.add_nodes_from(faculty.courses)
    takers: dict[int, list[str]] = defaultdict(list)
    for course, students in faculty.enrolment.items():
        for s in students:
            takers[s].append(course)
    for courses in takers.values():
        for a, b in combinations(sorted(courses), 2):
            if G.has_edge(a, b):
                G[a][b]["students"] += 1
            else:
                G.add_edge(a, b, students=1, lecturer=False)
    by_lecturer: dict[str, list[str]] = defaultdict(list)
    for course, lect in faculty.lecturer.items():
        by_lecturer[lect].append(course)
    for courses in by_lecturer.values():
        for a, b in combinations(sorted(courses), 2):
            if G.has_edge(a, b):
                G[a][b]["lecturer"] = True
            else:
                G.add_edge(a, b, students=0, lecturer=True)
    return G


def slot_name(slot: int, blocks_per_day: int = 6) -> str:
    """Human-readable name for a slot index, e.g. ``"Tue-2"`` (second block
    on Tuesday). Slots beyond Friday are labelled ``"Extra1-1"``, etc."""
    day, block = divmod(slot, blocks_per_day)
    name = DAYS[day] if day < len(DAYS) else f"Extra{day - len(DAYS) + 1}"
    return f"{name}-{block + 1}"


def timetable(coloring: dict[str, int], blocks_per_day: int = 6) -> dict[str, list[str]]:
    """Group courses by slot: ``{"Mon-1": [...], ...}`` in slot order."""
    slots: dict[int, list[str]] = defaultdict(list)
    for course, slot in coloring.items():
        slots[slot].append(course)
    return {slot_name(s, blocks_per_day): sorted(slots[s]) for s in sorted(slots)}


def clashes(G: nx.Graph, coloring: dict[str, int]) -> int:
    """Number of students with two courses in the same slot, plus lecturer
    double bookings. Zero for a valid timetable."""
    total = 0
    for a, b, data in G.edges(data=True):
        if coloring[a] == coloring[b]:
            total += data["students"] + int(data["lecturer"])
    return total
