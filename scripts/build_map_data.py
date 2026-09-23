"""Build the map datasets shipped in ``graph_coloring/apps/data``.

Downloads Natural Earth (public domain) boundaries and the communes of the
Santiago Metropolitan Region (official boundaries from the Biblioteca del
Congreso Nacional de Chile, via github.com/caracena/chile-geojson), computes which regions
share a border of positive length (touching at a single point, like the
US "Four Corners", does not count), simplifies the geometries so the files
stay small, and writes one GeoJSON per map with the adjacency list stored
in the top-level ``"adjacency"`` member.

Requires ``shapely`` (not needed at runtime):

    pip install shapely
    python scripts/build_map_data.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from shapely import make_valid
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
OUT = Path(__file__).resolve().parents[1] / "src" / "graph_coloring" / "apps" / "data"
CACHE = Path(__file__).resolve().parent / ".cache"
SANTIAGO = "https://raw.githubusercontent.com/caracena/chile-geojson/master/13.geojson"

# Minimum shared border length (in degrees) for two regions to be adjacent.
# Borders are measured with a 1e-3 degree buffer, so a single touching point
# shows up as about 2e-3 degrees; real borders are much longer.
MIN_BORDER = 0.01

# Natural Earth extends some US states into the Great Lakes, which creates
# water-only "borders". They are not visible on a map, so we drop them.
WATER_BORDERS = {("Illinois", "Michigan"), ("Michigan", "Minnesota")}


def load(name: str, url: str | None = None) -> dict:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if not path.exists():
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url or NE + name, path)
    return json.loads(path.read_text(encoding="utf-8"))


def adjacency(geoms: dict[str, object], min_border: float = MIN_BORDER) -> list[list[str]]:
    names = sorted(geoms)
    edges = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ga, gb = geoms[a], geoms[b]
            if not ga.buffer(1e-3).intersects(gb):
                continue
            shared = ga.buffer(1e-3).intersection(gb.boundary).length
            if shared >= min_border and (a, b) not in WATER_BORDERS:
                edges.append([a, b])
    return edges


def write(
    name: str,
    geoms: dict[str, object],
    tolerance: float,
    title: str,
    min_lon: float = -180,
    source: str = "Natural Earth (public domain), https://www.naturalearthdata.com",
    min_border: float = MIN_BORDER,
) -> None:
    """Write a map. Parts west of ``min_lon`` (remote islands) are dropped
    from the drawing to keep it compact; adjacency uses the full shapes."""
    edges = adjacency(geoms, min_border)
    features = []
    frame = box(min_lon, -90, 180, 90)
    for region, geom in sorted(geoms.items()):
        simple = geom.intersection(frame).simplify(tolerance, preserve_topology=True)
        features.append(
            {
                "type": "Feature",
                "properties": {"name": region},
                "geometry": _round(mapping(simple)),
            }
        )
    data = {
        "type": "FeatureCollection",
        "name": title,
        "source": source,
        "adjacency": edges,
        "features": features,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.geojson"
    path.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(
        f"{path.name}: {len(features)} regions, {len(edges)} borders, "
        f"{path.stat().st_size / 1024:.0f} KiB"
    )


def _round(geom: dict, nd: int = 3) -> dict:
    def r(x):
        return [r(y) for y in x] if isinstance(x[0], list | tuple) else [round(v, nd) for v in x]

    return {"type": geom["type"], "coordinates": r(geom["coordinates"])}


def south_america() -> None:
    countries = load("ne_50m_admin_0_countries.geojson")
    geoms = {}
    for f in countries["features"]:
        p = f["properties"]
        if p["CONTINENT"] == "South America" and p["NAME"] != "Falkland Is.":
            geoms[p["NAME"]] = shape(f["geometry"])
        elif p["NAME"] == "France":
            # French Guiana is the part of France inside South America
            # (Martinique and Guadeloupe lie further north, in the Caribbean).
            parts = [
                g
                for g in shape(f["geometry"]).geoms
                if -56 < g.centroid.x < -50 and g.centroid.y < 7
            ]
            geoms["French Guiana"] = unary_union(parts)
    # min_lon drops the Galápagos Islands
    write("south_america", geoms, 0.05, "South America", min_lon=-85)


def admin1(
    country: str, name: str, title: str, tolerance: float, exclude=(), min_lon: float = -180
) -> None:
    provinces = load("ne_10m_admin_1_states_provinces.geojson")
    geoms = {
        f["properties"]["name"]: shape(f["geometry"])
        for f in provinces["features"]
        if f["properties"]["adm0_a3"] == country and f["properties"]["name"] not in exclude
    }
    write(name, geoms, tolerance, title, min_lon)


def santiago() -> None:
    data = load("santiago_13.geojson", SANTIAGO)
    geoms = {f["properties"]["Comuna"]: make_valid(shape(f["geometry"])) for f in data["features"]}
    write(
        "santiago_communes",
        geoms,
        0.002,
        "Communes of the Santiago Metropolitan Region",
        source="Biblioteca del Congreso Nacional de Chile, via github.com/caracena/chile-geojson",
        # communes are small: keep borders from ~0.45 km, drop 4-way corners
        min_border=0.004,
    )


if __name__ == "__main__":
    south_america()
    # min_lon drops Easter Island and the Juan Fernández Islands
    admin1("CHL", "chile_regions", "Regions of Chile", 0.02, min_lon=-76)
    admin1(
        "USA",
        "us_states",
        "Contiguous United States",
        0.03,
        exclude={"Alaska", "Hawaii", "District of Columbia"},
    )
    santiago()
