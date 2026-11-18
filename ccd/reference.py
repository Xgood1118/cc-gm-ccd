# -*- coding: utf-8 -*-
"""Independent reference times, used only for verification and reporting.

The engine in :mod:`ccd.core` enumerates contact features and solves
quadratics.  The oracle here deliberately uses a *different* algorithm:
conservative advancement on the signed distance function.  Signed distance to
a convex set is 1-Lipschitz along a straight path, so ``g(t) / |v|`` is a
guaranteed safe advance, and the iteration converges to the first contact
without ever stepping past it.  That makes it a genuine cross-check rather
than a re-typed copy of the engine.

The oracle is not part of the public collision API.
"""

from __future__ import annotations

import math

from .core import _normalise_target

__all__ = ["reference_time", "signed_distance"]

_MAX_ITERATIONS = 20000


def _dist_point_segment(p, a, b):
    d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    if dd == 0.0:
        return math.dist(p, a)
    s = (
        (p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1] + (p[2] - a[2]) * d[2]
    ) / dd
    s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    q = (a[0] + d[0] * s, a[1] + d[1] * s, a[2] + d[2] * s)
    return math.dist(p, q)


def _signed_distance_shape(p, shape, inflate):
    kind = shape[0]
    if kind == "sphere":
        return math.dist(p, shape[1]) - (shape[2] + inflate)
    if kind == "capsule":
        return _dist_point_segment(p, shape[1], shape[2]) - (shape[3] + inflate)
    if kind == "box":
        center, half, rotation = shape[1], shape[2], shape[3]
        w = [p[i] - center[i] for i in range(3)]
        if rotation is None:
            q = w
        else:
            q = [sum(rotation[i][j] * w[j] for j in range(3)) for i in range(3)]
        outside = [max(0.0, abs(q[i]) - half[i]) for i in range(3)]
        return math.sqrt(sum(x * x for x in outside)) - inflate
    if kind == "compound":
        return min(_signed_distance_shape(p, part, inflate) for part in shape[1])
    raise TypeError("reference oracle does not support %r" % (kind,))


def signed_distance(p, target):
    """Signed distance from point ``p`` to the surface of ``target``."""
    return _signed_distance_shape(p, _normalise_target(target), 0.0)


def reference_time(start, velocity, target, dt, radius):
    """First contact time in ``[0, dt]`` or ``None`` (conservative oracle)."""
    start = tuple(float(x) for x in start)
    velocity = tuple(float(x) for x in velocity)
    dt = float(dt)
    radius = float(radius)
    shape = _normalise_target(target)

    speed = math.sqrt(sum(x * x for x in velocity))
    scale = max(
        1.0,
        radius,
        abs(dt),
        math.dist(start, (0.0, 0.0, 0.0)),
        speed * abs(dt),
    )
    # Geometric tolerance: contact is declared when the gap this small.
    gap_tol = 1e-12 * scale

    def gap(t):
        p = tuple(start[i] + velocity[i] * t for i in range(3))
        return _signed_distance_shape(p, shape, radius)

    if gap(0.0) <= gap_tol:
        return 0.0
    if speed == 0.0 or dt <= 0.0:
        return None

    t = 0.0
    for _ in range(_MAX_ITERATIONS):
        g = gap(t)
        if g <= gap_tol:
            return t
        step = g / speed
        if t + step > dt:
            return None
        t += step
    # Pathological convergence (e.g. an exact parallel graze): report the
    # limiting time rather than a sentinel.
    return t if t <= dt else None
