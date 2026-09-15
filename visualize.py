"""Interactive 3D visualization of flagged conjunctions.

Reads ``conjunction_alerts.csv``, draws Earth as a wireframe globe, plots every
flagged conjunction at its TEME Cartesian position, and writes a standalone
``index.html`` that needs no server and no network.

    python visualize.py
    python visualize.py --limit 20000 --output docs/index.html
    python visualize.py --color-by altitude_km

A note on point count: a full 72-hour screen produces ~184,000 encounters, and
handing all of them to a browser makes a file hundreds of MB that scrolls at a
crawl. The default plots the closest ``--limit`` encounters, which are the ones
that matter, and says so on the figure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

EARTH_RADIUS_KM = 6378.137

# Dark palette: the plot reads as space, and the colour scale stays legible
# against it.
BACKGROUND = "#05070d"
EARTH_SURFACE = [[0.0, "#0b2545"], [0.5, "#13395e"], [1.0, "#1d5b8f"]]
GRATICULE = "rgba(120, 190, 255, 0.35)"


def load_alerts(path: Path) -> pd.DataFrame:
    """Load the alert CSV, failing with a useful message rather than a stack."""
    if not path.exists():
        raise SystemExit(
            f"No alert file at {path}.\n"
            "Run the pipeline first, e.g.:  python main.py --full"
        )

    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(
            f"{path} contains no conjunctions. Nothing to plot.\n"
            "Try a longer window or a wider threshold, e.g.:  "
            "python main.py --full --threshold 10"
        )

    required = {"x_km", "y_km", "z_km", "miss_distance_km"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(
            f"{path} is missing required columns: {', '.join(sorted(missing))}.\n"
            "It was probably written by an older build; re-run the pipeline."
        )

    # Rows without a position cannot be placed in 3D.
    frame = frame.dropna(subset=["x_km", "y_km", "z_km"])
    if frame.empty:
        raise SystemExit(f"{path} has no rows with usable coordinates.")
    return frame


def earth_surface(resolution: int = 64) -> go.Surface:
    """A shaded sphere at Earth's mean equatorial radius."""
    lon = np.linspace(0, 2 * np.pi, resolution)
    lat = np.linspace(0, np.pi, resolution // 2)
    x = EARTH_RADIUS_KM * np.outer(np.cos(lon), np.sin(lat))
    y = EARTH_RADIUS_KM * np.outer(np.sin(lon), np.sin(lat))
    z = EARTH_RADIUS_KM * np.outer(np.ones_like(lon), np.cos(lat))

    return go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale=EARTH_SURFACE,
        surfacecolor=z,
        showscale=False,
        opacity=1.0,
        hoverinfo="skip",
        lighting=dict(ambient=0.65, diffuse=0.55, specular=0.08, roughness=0.9),
        name="Earth",
        showlegend=False,
    )


def earth_wireframe(meridians: int = 12, parallels: int = 6) -> list[go.Scatter3d]:
    """Graticule lines, drawn just above the surface so they are not z-fought."""
    radius = EARTH_RADIUS_KM * 1.002
    traces: list[go.Scatter3d] = []
    arc = np.linspace(0, 2 * np.pi, 180)

    for lon in np.linspace(0, np.pi, meridians, endpoint=False):
        traces.append(
            go.Scatter3d(
                x=radius * np.cos(lon) * np.sin(arc),
                y=radius * np.sin(lon) * np.sin(arc),
                z=radius * np.cos(arc),
                mode="lines",
                line=dict(color=GRATICULE, width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for lat in np.linspace(-np.pi / 2, np.pi / 2, parallels + 2)[1:-1]:
        traces.append(
            go.Scatter3d(
                x=radius * np.cos(lat) * np.cos(arc),
                y=radius * np.cos(lat) * np.sin(arc),
                z=np.full_like(arc, radius * np.sin(lat)),
                mode="lines",
                line=dict(color=GRATICULE, width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return traces


def _hover_text(frame: pd.DataFrame) -> list[str]:
    """One hover label per conjunction, naming both objects."""
    def field(name: str, default=""):
        return frame[name] if name in frame.columns else pd.Series(
            [default] * len(frame), index=frame.index
        )

    name_1, name_2 = field("name_1", "?"), field("name_2", "?")
    norad_1, norad_2 = field("norad_1", "?"), field("norad_2", "?")
    tca = field("tca_utc", "?")
    rel = field("relative_speed_km_s", float("nan"))
    alt = field("altitude_km", float("nan"))

    return [
        (
            f"<b>{n1}</b> ({i1})<br>"
            f"<b>{n2}</b> ({i2})<br>"
            f"TCA: {t} UTC<br>"
            f"Miss distance: {m:.3f} km<br>"
            f"Relative speed: {v:.3f} km/s<br>"
            f"Altitude: {a:,.0f} km"
        )
        for n1, i1, n2, i2, t, m, v, a in zip(
            name_1, norad_1, name_2, norad_2, tca,
            frame["miss_distance_km"], rel, alt,
        )
    ]


def conjunction_scatter(frame: pd.DataFrame, color_by: str) -> go.Scatter3d:
    """The flagged conjunctions, coloured by risk (tighter miss = hotter)."""
    values = frame[color_by]
    # Closest approaches are the dangerous ones, so reverse the scale: low miss
    # distance should read as hot, not cold.
    reverse = color_by == "miss_distance_km"

    return go.Scatter3d(
        x=frame["x_km"],
        y=frame["y_km"],
        z=frame["z_km"],
        mode="markers",
        marker=dict(
            size=2.6,
            color=values,
            colorscale="Inferno" if reverse else "Viridis",
            reversescale=reverse,
            opacity=0.85,
            colorbar=dict(
                title=dict(
                    text=color_by.replace("_", " "), side="right"
                ),
                thickness=14,
                len=0.6,
                tickfont=dict(color="#c9d6e8"),
            ),
        ),
        text=_hover_text(frame),
        hovertemplate="%{text}<extra></extra>",
        name="Conjunctions",
        showlegend=False,
    )


def build_figure(
    frame: pd.DataFrame, total_rows: int, color_by: str, source: Path
) -> go.Figure:
    traces: list[go.BaseTraceType] = [earth_surface()]
    traces.extend(earth_wireframe())
    traces.append(conjunction_scatter(frame, color_by))

    shown = len(frame)
    if shown < total_rows:
        subtitle = (
            f"Closest {shown:,} of {total_rows:,} flagged encounters "
            f"· {source.name}"
        )
    else:
        subtitle = f"{shown:,} flagged encounters · {source.name}"

    closest = frame["miss_distance_km"].min()
    subtitle += f" · tightest approach {closest:.3f} km"

    # `aspectmode="data"` keeps Earth spherical; without it Plotly stretches the
    # axes to the box and the globe becomes an ellipsoid.
    axis = dict(
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title="",
    )

    figure = go.Figure(data=traces)
    figure.update_layout(
        title=dict(
            text=(
                "<b>Project Orbital Sentinel</b><br>"
                f"<span style='font-size:13px;color:#8fa6c4'>{subtitle}</span>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(color="#e8eef8", size=22),
        ),
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        scene=dict(
            xaxis=axis,
            yaxis=axis,
            zaxis=axis,
            aspectmode="data",
            bgcolor=BACKGROUND,
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=90, b=0),
        hoverlabel=dict(
            bgcolor="#0f1626",
            bordercolor="#2b3d5c",
            font=dict(color="#e8eef8", size=12),
        ),
    )
    return figure


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="visualize",
        description="Render flagged conjunctions as an interactive 3D globe.",
    )
    parser.add_argument(
        "--input", default="conjunction_alerts.csv",
        help="Alert CSV to read (default: conjunction_alerts.csv).",
    )
    parser.add_argument(
        "--output", default="index.html",
        help="HTML file to write (default: index.html).",
    )
    parser.add_argument(
        "--limit", type=int, default=15000,
        help="Plot only the closest N encounters; 0 plots all. Large values "
             "make the page slow to open (default: 15000).",
    )
    parser.add_argument(
        "--color-by", default="miss_distance_km",
        choices=["miss_distance_km", "relative_speed_km_s", "altitude_km"],
        help="Column driving the colour scale (default: miss_distance_km).",
    )
    parser.add_argument(
        "--cdn", action="store_true",
        help="Load plotly.js from a CDN instead of inlining it. Shrinks the "
             "page from ~9 MB to ~1 MB, but it then needs a network to open.",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open the rendered page in a browser when done.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = Path(args.input)
    frame = load_alerts(source)
    total_rows = len(frame)

    if args.limit and total_rows > args.limit:
        frame = frame.nsmallest(args.limit, "miss_distance_km")
        print(
            f"Plotting the closest {len(frame):,} of {total_rows:,} encounters "
            f"(raise with --limit, or --limit 0 for all)."
        )

    figure = build_figure(frame, total_rows, args.color_by, source)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Inlining plotly.js keeps the page working offline and behind a firewall,
    # at the cost of a few MB; --cdn trades that for a much smaller file.
    figure.write_html(
        str(output),
        include_plotlyjs="cdn" if args.cdn else "inline",
        full_html=True,
        config={"displaylogo": False, "scrollZoom": True},
    )

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.1f} MB, {len(frame):,} points).")

    if args.open:
        import webbrowser

        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
