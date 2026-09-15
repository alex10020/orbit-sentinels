"""Animated 3D orbital simulation with play/pause controls.

Where ``visualize.py`` plots *where* conjunctions happen as a static cloud,
this renders the catalog actually moving: objects propagate along their orbits
frame by frame, and conjunctions ignite as they occur.

    python simulate.py --open
    python simulate.py --objects 1500 --minutes 95 --fps-steps 180
    python simulate.py --highlight-top 25 --open

Every frame is precomputed with the same vectorized SGP4 engine the pipeline
uses, then handed to Plotly as animation frames, so the browser only replays
positions -- it does no orbital mechanics of its own.

Frame budget matters: the page holds ``objects x frames x 3`` floats in memory.
The defaults (1,200 objects, 150 frames) stay comfortable in a browser; pushing
far past that makes the page sluggish regardless of how fast the physics was.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import config
from ingestion import load_catalog
from logger import get_logger
from propagation import PropagationEngine, build_time_grid
from spatial_index import detect_pairs

log = get_logger("simulate")

BACKGROUND = "#05070d"
EARTH_SURFACE = [[0.0, "#0b2545"], [0.5, "#13395e"], [1.0, "#1d5b8f"]]
GRATICULE = "rgba(120, 190, 255, 0.30)"
OBJECT_COLOR = "#7fd4ff"
CONJUNCTION_COLOR = "#ff3b30"


def earth_traces(resolution: int = 48) -> list:
    """Shaded globe plus a graticule, matching visualize.py."""
    lon = np.linspace(0, 2 * np.pi, resolution)
    lat = np.linspace(0, np.pi, resolution // 2)
    radius = config.EARTH_RADIUS_KM

    surface = go.Surface(
        x=radius * np.outer(np.cos(lon), np.sin(lat)),
        y=radius * np.outer(np.sin(lon), np.sin(lat)),
        z=radius * np.outer(np.ones_like(lon), np.cos(lat)),
        colorscale=EARTH_SURFACE,
        surfacecolor=radius * np.outer(np.ones_like(lon), np.cos(lat)),
        showscale=False,
        hoverinfo="skip",
        lighting=dict(ambient=0.65, diffuse=0.55, specular=0.08, roughness=0.9),
        showlegend=False,
    )

    wire: list = []
    arc = np.linspace(0, 2 * np.pi, 160)
    r = radius * 1.002
    for value in np.linspace(0, np.pi, 12, endpoint=False):
        wire.append(
            go.Scatter3d(
                x=r * np.cos(value) * np.sin(arc),
                y=r * np.sin(value) * np.sin(arc),
                z=r * np.cos(arc),
                mode="lines",
                line=dict(color=GRATICULE, width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    for value in np.linspace(-np.pi / 2, np.pi / 2, 8)[1:-1]:
        wire.append(
            go.Scatter3d(
                x=r * np.cos(value) * np.cos(arc),
                y=r * np.cos(value) * np.sin(arc),
                z=np.full_like(arc, r * np.sin(value)),
                mode="lines",
                line=dict(color=GRATICULE, width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return [surface] + wire


def simulate(
    n_objects: int,
    minutes: float,
    n_frames: int,
    threshold_km: float,
    min_rel_speed_kms: float,
) -> dict:
    """Propagate a subset of the catalog and find conjunctions per frame.

    Returns the per-frame positions, the conjunctions detected in each frame,
    and the screening statistics shown in the on-screen counters.
    """
    objects = load_catalog(max_objects=n_objects)
    if len(objects) < 2:
        raise SystemExit("Need at least two objects to simulate.")

    engine = PropagationEngine(objects)
    step_minutes = minutes / n_frames
    start = datetime.now(timezone.utc)
    grid = build_time_grid(start=start, hours=minutes / 60.0, step_minutes=step_minutes)

    log.info(
        "Propagating %d objects across %d frames (%.2f min steps, %.0f min span).",
        len(engine), len(grid), step_minutes, minutes,
    )
    result = engine.propagate(grid)

    frames: list[dict] = []
    total_detected = 0
    for step in range(result.n_steps):
        pos, vel, valid = result.step(step)
        detection = detect_pairs(
            pos, vel, valid,
            threshold_km=threshold_km,
            min_rel_speed_kms=min_rel_speed_kms,
            step_minutes=step_minutes,
        )

        live = np.flatnonzero(valid)
        conj_xyz = np.empty((0, 3))
        labels: list[str] = []
        if len(detection):
            conj_xyz = detection.midpoint_km
            labels = [
                f"<b>{engine.names[int(detection.index_i[k])]}</b> &times; "
                f"<b>{engine.names[int(detection.index_j[k])]}</b><br>"
                f"Miss {detection.miss_km[k]:.3f} km &nbsp; "
                f"Rel {detection.rel_speed_kms[k]:.2f} km/s"
                for k in range(len(detection))
            ]
            total_detected += len(detection)

        frames.append(
            {
                "xyz": np.ascontiguousarray(pos[live]),
                "conj": conj_xyz,
                "labels": labels,
                "time": grid.timestamp(step),
                "n_live": int(live.size),
            }
        )

    n_live = int(np.median([f["n_live"] for f in frames]))
    brute_force = n_live * (n_live - 1) / 2
    return {
        "frames": frames,
        "n_objects": len(engine),
        "brute_force_per_frame": brute_force,
        "total_detected": total_detected,
        "step_minutes": step_minutes,
        "start": start,
    }


def build_figure(sim: dict, threshold_km: float, frame_ms: int) -> go.Figure:
    frames_data = sim["frames"]
    base = frames_data[0]

    static = earth_traces()
    n_static = len(static)

    # Trace order is fixed: [Earth..., objects, conjunctions]. Animation frames
    # target the last two by index, so the globe is never re-sent per frame.
    objects_trace = go.Scatter3d(
        x=base["xyz"][:, 0], y=base["xyz"][:, 1], z=base["xyz"][:, 2],
        mode="markers",
        marker=dict(size=1.9, color=OBJECT_COLOR, opacity=0.75),
        hoverinfo="skip",
        name="Catalog objects",
        showlegend=False,
    )
    conj_trace = go.Scatter3d(
        x=base["conj"][:, 0] if len(base["conj"]) else [],
        y=base["conj"][:, 1] if len(base["conj"]) else [],
        z=base["conj"][:, 2] if len(base["conj"]) else [],
        mode="markers",
        marker=dict(
            size=11,
            color=CONJUNCTION_COLOR,
            opacity=0.95,
            symbol="circle",
            line=dict(color="#ffd7d4", width=2),
        ),
        text=base["labels"],
        hovertemplate="%{text}<extra></extra>",
        name="Conjunction",
        showlegend=False,
    )

    plotly_frames = []
    for index, frame in enumerate(frames_data):
        conj = frame["conj"]
        plotly_frames.append(
            go.Frame(
                name=str(index),
                data=[
                    go.Scatter3d(
                        x=frame["xyz"][:, 0],
                        y=frame["xyz"][:, 1],
                        z=frame["xyz"][:, 2],
                    ),
                    go.Scatter3d(
                        x=conj[:, 0] if len(conj) else [],
                        y=conj[:, 1] if len(conj) else [],
                        z=conj[:, 2] if len(conj) else [],
                        text=frame["labels"],
                    ),
                ],
                traces=[n_static, n_static + 1],
                layout=go.Layout(
                    annotations=_counters(frame, sim, threshold_km)
                ),
            )
        )

    figure = go.Figure(
        data=static + [objects_trace, conj_trace], frames=plotly_frames
    )

    axis = dict(
        showbackground=False, showgrid=False, zeroline=False,
        showticklabels=False, title="",
    )
    figure.update_layout(
        title=dict(
            text=(
                "<b>Orbital Sentinel &mdash; Live Propagation</b><br>"
                "<span style='font-size:13px;color:#8fa6c4'>"
                f"{sim['n_objects']:,} objects · vectorized SGP4 · "
                f"cKDTree screen at {threshold_km:.0f} km</span>"
            ),
            x=0.5, xanchor="center",
            font=dict(color="#e8eef8", size=22),
        ),
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        scene=dict(
            xaxis=axis, yaxis=axis, zaxis=axis,
            aspectmode="data",
            bgcolor=BACKGROUND,
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.85)),
        ),
        margin=dict(l=0, r=0, t=95, b=10),
        annotations=_counters(frames_data[0], sim, threshold_km),
        hoverlabel=dict(
            bgcolor="#0f1626", bordercolor="#2b3d5c",
            font=dict(color="#e8eef8", size=12),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.5, xanchor="center",
                y=0.04, yanchor="bottom",
                bgcolor="#16213a",
                bordercolor="#2b3d5c",
                font=dict(color="#e8eef8", size=13),
                pad=dict(l=8, r=8, t=6, b=6),
                buttons=[
                    dict(
                        label="  Play  ",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_ms, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="  Pause  ",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.08, len=0.84,
                y=0.0, yanchor="bottom",
                pad=dict(t=4, b=4),
                bgcolor="#1d2740",
                bordercolor="#2b3d5c",
                font=dict(color="#c9d6e8", size=11),
                currentvalue=dict(
                    prefix="T+ ",
                    font=dict(color="#e8eef8", size=13),
                    visible=True,
                ),
                steps=[
                    dict(
                        method="animate",
                        label=f"{i * sim['step_minutes']:.0f} min",
                        args=[
                            [str(i)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for i in range(len(frames_data))
                ],
            )
        ],
    )
    return figure


def _counters(frame: dict, sim: dict, threshold_km: float) -> list[dict]:
    """On-screen readout: what the spatial index saved on this frame."""
    n_conj = len(frame["conj"])
    brute = sim["brute_force_per_frame"]
    saved = 100.0 * (1.0 - (n_conj / brute)) if brute else 0.0

    def cell(x: float, title: str, value: str, color: str = "#e8eef8") -> list[dict]:
        return [
            dict(
                x=x, y=0.17, xref="paper", yref="paper",
                text=f"<span style='font-size:11px;color:#7d90ac'>{title}</span>",
                showarrow=False, align="center",
            ),
            dict(
                x=x, y=0.12, xref="paper", yref="paper",
                text=f"<b style='font-size:19px;color:{color}'>{value}</b>",
                showarrow=False, align="center",
            ),
        ]

    out: list[dict] = []
    out += cell(0.16, "SIMULATION TIME", frame["time"].strftime("%H:%M:%S") + "Z")
    out += cell(0.38, "OBJECTS PROPAGATED", f"{frame['n_live']:,}")
    out += cell(0.62, "CONJUNCTIONS NOW",
                str(n_conj), CONJUNCTION_COLOR if n_conj else "#e8eef8")
    out += cell(0.85, "PAIR CHECKS SAVED", f"{saved:.2f}%", "#6fe3a1")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="simulate",
        description="Animated 3D orbital simulation with conjunction flashes.",
    )
    parser.add_argument(
        "--objects", type=int, default=1200,
        help="Objects to animate (default: 1200). Large values slow the page.",
    )
    parser.add_argument(
        "--minutes", type=float, default=95.0,
        help="Simulated span; 95 min is about one LEO orbit (default: 95).",
    )
    parser.add_argument(
        "--frames", type=int, default=150,
        help="Animation frames across the span (default: 150).",
    )
    parser.add_argument(
        "--threshold", type=float, default=25.0,
        help="Screening radius in km. Wider than the 5 km operational "
             "threshold so the animation actually shows events (default: 25).",
    )
    parser.add_argument(
        "--min-rel-speed", type=float, default=config.MIN_RELATIVE_SPEED_KM_S,
        help="Co-orbiting filter, as in the main pipeline.",
    )
    parser.add_argument(
        "--frame-ms", type=int, default=60,
        help="Milliseconds per frame when playing (default: 60).",
    )
    parser.add_argument(
        "--output", default="simulation.html", help="HTML file to write.",
    )
    parser.add_argument(
        "--cdn", action="store_true",
        help="Load plotly.js from a CDN instead of inlining it.",
    )
    parser.add_argument("--open", action="store_true", help="Open when done.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sim = simulate(
        n_objects=args.objects,
        minutes=args.minutes,
        n_frames=args.frames,
        threshold_km=args.threshold,
        min_rel_speed_kms=args.min_rel_speed,
    )
    log.info(
        "Detected %d conjunction instances across %d frames.",
        sim["total_detected"], len(sim["frames"]),
    )

    figure = build_figure(sim, args.threshold, args.frame_ms)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        str(output),
        include_plotlyjs="cdn" if args.cdn else "inline",
        full_html=True,
        auto_play=False,
        config={"displaylogo": False, "scrollZoom": True},
    )
    size_mb = output.stat().st_size / (1024 * 1024)
    log.info("Wrote %s (%.1f MB, %d frames).", output, size_mb, len(sim["frames"]))

    if args.open:
        import webbrowser

        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
