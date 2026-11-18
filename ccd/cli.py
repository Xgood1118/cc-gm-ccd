# -*- coding: utf-8 -*-
"""Command line runner.

    python -m ccd.cli data/queries.json --out hits.json

Every target in the file is run through :func:`ccd.time_of_impact`; the
resulting hits are written to ``--out`` (when given) and a summary with the
query count, hit count, largest time error and tangent-hit count is printed.
The input file is only ever read.
"""

from __future__ import annotations

import argparse
import json
import time

import ccd as m

from .reference import reference_time

__all__ = ["evaluate", "main"]

DEFAULT_DT = 3.0
DEFAULT_RADIUS = 0.5


def _hits_to_json(hits):
    out = []
    for t, target in hits:
        depth = getattr(t, "depth", 0.0)
        tangent = bool(getattr(t, "tangent", False))
        out.append(
            {
                "time": float(t),
                "depth": float(depth),
                "tangent": tangent,
                "target": _describe(target),
            }
        )
    return out


def _describe(target):
    if isinstance(target, m.Sphere):
        return {"type": "sphere", "center": list(target.center), "radius": target.radius}
    if isinstance(target, m.Capsule):
        return {
            "type": "capsule",
            "a": list(target.a),
            "b": list(target.b),
            "radius": target.radius,
        }
    if isinstance(target, m.Box):
        return {
            "type": "box",
            "center": list(target.center),
            "half_extents": list(target.half_extents),
            "rotation": None if target.rotation is None else [list(r) for r in target.rotation],
        }
    if isinstance(target, dict):
        return target
    return {"repr": repr(target)}


def evaluate(document, dt=DEFAULT_DT, radius=DEFAULT_RADIUS, check_reference=False):
    """Run the whole document and collect summary statistics.

    ``check_reference`` additionally compares every result against the
    independent oracle in :mod:`ccd.reference`; ``max_time_error`` is only
    meaningful in that mode (it is ``None`` otherwise).
    """
    queries = document.get("queries", [])
    n_targets = 0
    hit_rows = []
    tangent_hits = 0
    zero_hits = 0
    max_error_reference = 0.0
    mismatches = 0

    started = time.perf_counter()
    for q in queries:
        start = q["start"]
        velocity = q["velocity"]
        targets = q["targets"]
        n_targets += len(targets)
        hits = m.time_of_impact(start, velocity, targets, dt, radius)
        for t, target in hits:
            hit_rows.append((t, target))
            if getattr(t, "tangent", False):
                tangent_hits += 1
            if float(t) == 0.0:
                zero_hits += 1
        if check_reference:
            for target in targets:
                ref = reference_time(start, velocity, target, dt, radius)
                hit = m.sweep_sphere(start, velocity, target, dt, radius)
                if (ref is None) != (hit is None):
                    mismatches += 1
                    continue
                if ref is not None:
                    err = abs(float(hit) - ref)
                    if err > max_error_reference:
                        max_error_reference = err
    elapsed = time.perf_counter() - started

    return {
        "queries": len(queries),
        "targets": n_targets,
        "hits": len(hit_rows),
        "tangent_hits": tangent_hits,
        "zero_time_hits": zero_hits,
        "max_time_error": max_error_reference if check_reference else None,
        "reference_mismatches": mismatches if check_reference else None,
        "elapsed": elapsed,
        "_rows": hit_rows,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="ccd")
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    ap.add_argument(
        "--check-reference",
        action="store_true",
        help="compare every hit against the independent reference oracle",
    )
    a = ap.parse_args(argv)

    with open(a.path, encoding="utf-8") as fh:
        document = json.load(fh)

    stats = evaluate(document, dt=a.dt, radius=a.radius, check_reference=a.check_reference)
    rows = stats.pop("_rows")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "queries": stats["queries"],
                    "targets": stats["targets"],
                    "hits": stats["hits"],
                    "tangent_hits": stats["tangent_hits"],
                    "zero_time_hits": stats["zero_time_hits"],
                    "max_time_error": stats["max_time_error"],
                    "elapsed": stats["elapsed"],
                    "hits_detail": _hits_to_json(rows),
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

    summary = {
        "queries": stats["queries"],
        "targets": stats["targets"],
        "hits": stats["hits"],
        "tangent_hits": stats["tangent_hits"],
        "zero_time_hits": stats["zero_time_hits"],
        "max_time_error": stats["max_time_error"],
        "elapsed": round(stats["elapsed"], 6),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
