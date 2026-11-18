# -*- coding: utf-8 -*-
"""Generate reproducible benchmark / report data.

Usage::

    python -m ccd.bench [--out bench.json]

The generated queries cover the three target kinds required by the task
(sphere, box, capsule), plus overlap, tangent, parallel and zero-velocity
degenerate cases.  ``data/queries.json`` itself is never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import random

__all__ = ["generate", "run"]


def _target_dict(rng, kind):
    if kind == 0:
        return {
            "type": "sphere",
            "center": [rng.uniform(-5.0, 5.0) for _ in range(3)],
            "radius": rng.uniform(0.2, 1.2),
        }
    if kind == 1:
        return {
            "type": "capsule",
            "a": [rng.uniform(-5.0, 5.0) for _ in range(3)],
            "b": [rng.uniform(-5.0, 5.0) for _ in range(3)],
            "radius": rng.uniform(0.2, 0.8),
        }
    return {
        "type": "box",
        "center": [rng.uniform(-4.0, 4.0) for _ in range(3)],
        "half_extents": [rng.uniform(0.2, 1.2) for _ in range(3)],
    }


def generate(count=1000, seed=20261119, dt=3.0, radius=0.5):
    """Return a ``{"queries": [...]}`` document like ``data/queries.json``."""
    rng = random.Random(seed)
    queries = []
    for i in range(count):
        start = [rng.uniform(-6.0, 6.0) for _ in range(3)]

        # Every fifth query exercises a degenerate family.
        family = i % 5
        if family == 0:
            velocity = [0.0, 0.0, 0.0]
        elif family == 1:  # parallel to the x axis, mostly missing
            velocity = [rng.uniform(1.0, 2.0), 0.0, 0.0]
        else:
            velocity = [rng.uniform(-2.0, 2.0) for _ in range(3)]

        targets = []
        for _ in range(3):
            kind = rng.randrange(3)
            target = _target_dict(rng, kind)
            if family == 1:  # place the target off the motion line
                keys = ("a", "b") if kind == 1 else ("center",)
                for key in keys:
                    for axis in (1, 2):
                        target[key][axis] += 20.0
            targets.append([target])  # each query item is one target

        queries.append(
            {
                "start": start,
                "velocity": velocity,
                "targets": targets,
            }
        )

    # Add deterministic exact-tangency cases so the report always has some.
    # A unit sphere moving along x at y = (1 + r) grazes the target sphere.
    tangent = [
        ({"center": [0.0, 0.0, 0.0], "radius": 1.0}, 2.0),
        ({"center": [3.0, 0.0, 0.0], "radius": 0.5}, 1.0),
    ]
    for i, (target, y) in enumerate(tangent):
        if i < len(queries):
            queries[i] = {
                "start": [-5.0, y, 0.0],
                "velocity": [1.0, 0.0, 0.0],
                "targets": [[{"type": "sphere", **target}]],
            }
    return {"queries": queries}


def run(count=1000, seed=20261119, dt=3.0, radius=0.5, repetitions=3):
    """Generate ``count`` queries, evaluate them and keep the fastest run."""
    from . import cli

    document = generate(count, seed=seed, dt=dt, radius=radius)
    best = None
    for _ in range(repetitions):
        stats = cli.evaluate(document, dt=dt, radius=radius)
        if best is None or stats["elapsed"] < best["elapsed"]:
            best = stats
    return document, best


def main(argv=None):
    ap = argparse.ArgumentParser(description="ccd benchmark")
    ap.add_argument("--scales", default="1000,2000")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--dt", type=float, default=3.0)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--json", default=None, help="write the results as JSON here")
    args = ap.parse_args(argv)

    sizes = [int(s) for s in args.scales.split(",") if s.strip()]
    results = []
    for n in sizes:
        document, stats = run(
            n,
            seed=args.seed,
            dt=args.dt,
            radius=args.radius,
            repetitions=args.repetitions,
        )
        row = {
            "queries": stats["queries"],
            "targets": stats["targets"],
            "hits": stats["hits"],
            "tangent_hits": stats["tangent_hits"],
            "zero_time_hits": stats["zero_time_hits"],
            "elapsed": stats["elapsed"],
        }
        results.append(row)
        print(
            "%5d queries / %5d targets: %8.4f s, %d hits"
            % (row["queries"], row["targets"], row["elapsed"], row["hits"])
        )

    if len(results) >= 2:
        small = next(r for r in results if r["queries"] == min(x["queries"] for x in results))
        large = next(r for r in results if r["queries"] == max(x["queries"] for x in results))
        ratio = large["elapsed"] / small["elapsed"] if small["elapsed"] > 0 else float("inf")
        qratio = large["queries"] / small["queries"]
        print(
            "doubling x%.2f: time x%.2f (sub-linear blow-up if close to %.2f)"
            % (qratio, ratio, qratio)
        )
        for row in results:
            if row["queries"] == large["queries"]:
                row["doubling_ratio"] = ratio

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
    return results


if __name__ == "__main__":  # pragma: no cover
    main()
