# -*- coding: utf-8 -*-
"""Continuous collision detection for a moving sphere.

The public API deliberately keeps the names and signatures of the original
skeleton::

    Sphere(center, radius)
    Capsule(a, b, radius)
    sweep_sphere(start, velocity, target, dt, radius) -> ImpactTime | None
    time_of_impact(start, velocity, targets, dt, radius) -> [(time, target)]

``Box`` (and the ``AABB`` convenience subclass) are additions, needed because
the task requires sphere/box sweeps.

Only the Python standard library is used.  All shapes are treated as static;
the sphere given by ``start``/``velocity``/``radius`` moves during ``[0, dt]``.

Design notes
------------
* First contact is found analytically.  Sphere/sphere, sphere/capsule (two
  spherical caps plus the cylindrical side) and sphere/box (six face patches,
  twelve edge cylinders and eight corner spheres of the rounded Minkowski
  sum) all have closed-form candidate times; the earliest valid candidate is
  the first contact.
* ``None`` is the one and only "no contact" marker.  Negative times and
  sentinel huge values are never returned.
* A hit at ``t == 0`` is returned as ``ImpactTime(0.0)``.  ``ImpactTime`` is a
  ``float`` subclass, so it behaves exactly like the old numeric return value,
  and additionally exposes ``.depth`` (penetration depth, >= 0),
  ``.tangent`` and ``.time``.  This is how "returns 0 and reports the overlap
  depth" is represented without changing the function signature.
* Degenerate inputs (already overlapping, zero velocity, motion parallel to a
  feature, exact tangency) are decided explicitly rather than by sampling.
  A relative tolerance makes exact tangency a stable, repeatable decision.
"""

from __future__ import annotations

import math

__all__ = [
    "AABB",
    "Box",
    "Capsule",
    "ImpactTime",
    "Sphere",
    "sweep_sphere",
    "time_of_impact",
]

# Relative tolerance used for tangency / boundary decisions.  It is applied to
# dimensionless quantities (quadratic discriminants scaled by their terms,
# region tests scaled by the local geometry), never as a blanket absolute
# epsilon on the returned time.
_EPS = 1e-12
# Tolerance used when comparing a returned time against the [0, dt] window.
_TIME_EPS = 1e-12


# ---------------------------------------------------------------------------
# small vector helpers
# ---------------------------------------------------------------------------


def _vec(v):
    """Return a 3-tuple of floats, with a clear error for malformed input."""
    try:
        return (float(v[0]), float(v[1]), float(v[2]))
    except (TypeError, IndexError, KeyError) as exc:  # pragma: no cover
        raise TypeError("expected a 3-component vector") from exc


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm2(a):
    return _dot(a, a)


def _norm(a):
    return math.sqrt(_dot(a, a))


def _clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ---------------------------------------------------------------------------
# public value types
# ---------------------------------------------------------------------------


class ImpactTime(float):
    """A first-contact time that can also carry collision metadata.

    It is a ``float`` subclass: ``t == 0.0``, ``abs(t - expected) < tol``,
    arithmetic, JSON encoding and normal float comparisons all keep working
    exactly as they did with a plain float.  Extra attributes:

    ``depth``
        Penetration depth at the contact.  Positive only when the start
        configuration already overlaps (``t == 0``); zero otherwise.
    ``tangent``
        True when contact is tangential (the discriminant is zero within the
        tolerance) rather than a transversal crossing.
    ``time``
        ``float(self)``, provided for readability.
    ``penetration`` / ``penetration_depth`` / ``overlap_depth`` / ``overlap``
        Aliases of ``depth``.
    ``target``
        The target object that produced this hit, when known.
    """

    def __new__(cls, value, depth=0.0, tangent=False, target=None):
        obj = super().__new__(cls, float(value))
        depth = float(depth)
        if depth < 0.0:
            depth = 0.0
        obj.depth = depth
        obj.tangent = bool(tangent)
        obj.target = target
        return obj

    # aliases -----------------------------------------------------------
    @property
    def time(self):
        return float(self)

    @property
    def penetration(self):
        return self.depth

    @property
    def penetration_depth(self):
        return self.depth

    @property
    def overlap_depth(self):
        return self.depth

    @property
    def overlap(self):
        return self.depth

    def __repr__(self):  # pragma: no cover - cosmetic
        return "ImpactTime(%r, depth=%r, tangent=%r)" % (
            float(self),
            self.depth,
            self.tangent,
        )


class Sphere:
    """A static sphere.

    Parameters
    ----------
    center : sequence of 3 numbers
    radius : non-negative number
    """

    __slots__ = ("center", "radius")

    def __init__(self, center, radius):
        self.center = _vec(center)
        self.radius = float(radius)
        if not math.isfinite(self.radius) or self.radius < 0.0:
            raise ValueError("Sphere radius must be a finite non-negative number")

    def __repr__(self):  # pragma: no cover - cosmetic
        return "Sphere(center=%r, radius=%r)" % (self.center, self.radius)


class Capsule:
    """A static capsule: the segment ``a``--``b`` inflated by ``radius``."""

    __slots__ = ("a", "b", "radius")

    def __init__(self, a, b, radius):
        self.a = _vec(a)
        self.b = _vec(b)
        self.radius = float(radius)
        if not math.isfinite(self.radius) or self.radius < 0.0:
            raise ValueError("Capsule radius must be a finite non-negative number")

    def __repr__(self):  # pragma: no cover - cosmetic
        return "Capsule(a=%r, b=%r, radius=%r)" % (self.a, self.b, self.radius)


class Box:
    """A static box (axis-aligned by default, optionally oriented).

    ``Box(center, half_extents)`` describes an axis-aligned box.
    ``rotation``, if given, is a 3x3 matrix whose *rows* are the box's local
    x/y/z axes expressed in world coordinates (so the world-to-local map is
    ``q = rotation @ (p - center)``).
    """

    __slots__ = ("center", "half_extents", "rotation")

    def __init__(self, center, half_extents, rotation=None):
        self.center = _vec(center)
        h = _vec(half_extents)
        if any((not math.isfinite(x)) for x in h) or any(x < 0.0 for x in h):
            raise ValueError("Box half extents must be finite non-negative numbers")
        self.half_extents = h
        self.rotation = _parse_rotation(rotation)

    @classmethod
    def from_bounds(cls, min_corner, max_corner):
        """Build an axis-aligned box from opposite corners."""
        lo = _vec(min_corner)
        hi = _vec(max_corner)
        center = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
        half = tuple(abs(hi[i] - lo[i]) * 0.5 for i in range(3))
        return cls(center, half)

    # aliases used by some callers -------------------------------------
    @classmethod
    def from_min_max(cls, min_corner, max_corner):
        return cls.from_bounds(min_corner, max_corner)

    @property
    def min_corner(self):
        return tuple(self.center[i] - self.half_extents[i] for i in range(3))

    @property
    def max_corner(self):
        return tuple(self.center[i] + self.half_extents[i] for i in range(3))

    @property
    def min(self):
        return self.min_corner

    @property
    def max(self):
        return self.max_corner

    def __repr__(self):  # pragma: no cover - cosmetic
        return "Box(center=%r, half_extents=%r, rotation=%r)" % (
            self.center,
            self.half_extents,
            self.rotation,
        )


class AABB(Box):
    """Convenience alias whose constructor takes opposite corners."""

    __slots__ = ()

    def __init__(self, min_corner, max_corner):
        lo = _vec(min_corner)
        hi = _vec(max_corner)
        center = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
        half = tuple(abs(hi[i] - lo[i]) * 0.5 for i in range(3))
        super().__init__(center, half)


def _parse_rotation(rotation):
    if rotation is None:
        return None
    if isinstance(rotation, (list, tuple)):
        rows = list(rotation)
        if len(rows) == 9 and all(isinstance(x, (int, float)) for x in rows):
            return tuple(
                (float(rows[3 * i]), float(rows[3 * i + 1]), float(rows[3 * i + 2]))
                for i in range(3)
            )
        if len(rows) == 3 and all(
            isinstance(r, (list, tuple)) and len(r) == 3 for r in rows
        ):
            return tuple(tuple(float(x) for x in r) for r in rows)
    raise ValueError("Box rotation must be a 3x3 matrix (or 9 numbers)")


# ---------------------------------------------------------------------------
# target normalisation
# ---------------------------------------------------------------------------


def _shape_from_dict(d):
    typ = d.get("type", d.get("shape", d.get("kind", d.get("primitive"))))
    typ = str(typ).lower() if typ is not None else None

    parts = d.get("shapes", d.get("parts", d.get("children")))
    if parts is not None or typ in ("compound", "union", "group"):
        if parts is None:
            raise ValueError("compound target needs 'shapes'/'parts'")
        return ("compound", tuple(_normalise_target(x) for x in parts))

    # capsule: endpoints + radius
    if typ in ("capsule", "segment") or ("a" in d and "b" in d):
        if "a" not in d or "b" not in d:
            raise ValueError("capsule target needs 'a', 'b' and 'radius'")
        rad = d.get("radius", d.get("r"))
        return ("capsule", _vec(d["a"]), _vec(d["b"]), float(rad))

    # box: center + half extents, or min/max bounds
    has_box_keys = any(
        k in d
        for k in (
            "half_extents",
            "half_extent",
            "half_size",
            "half",
            "extents",
            "size",
        )
    )
    if typ in ("box", "aabb", "obb", "cuboid") or has_box_keys or (
        ("min" in d or "min_corner" in d) and ("max" in d or "max_corner" in d)
    ):
        if "min" in d or "min_corner" in d:
            lo_key = "min" if "min" in d else "min_corner"
            hi_key = "max" if "max" in d else "max_corner"
            lo = _vec(d[lo_key])
            hi = _vec(d[hi_key])
            center = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
            half = tuple(abs(hi[i] - lo[i]) * 0.5 for i in range(3))
        else:
            center = _vec(d["center"])
            h = None
            for key in ("half_extents", "half_extent", "half_size", "half", "extents"):
                if key in d:
                    h = _vec(d[key])
                    break
            if h is None and "size" in d:
                s = _vec(d["size"])
                h = tuple(x * 0.5 for x in s)
            if h is None:
                raise ValueError("box target needs half extents or min/max bounds")
            half = h
        rot = _parse_rotation(
            d.get("rotation", d.get("orientation", d.get("rotation_matrix")))
        )
        return ("box", center, half, rot)

    # sphere: center + radius
    if typ in ("sphere", "ball") or ("center" in d and ("radius" in d or "r" in d)):
        center = _vec(d.get("center", d.get("c")))
        rad = d.get("radius", d.get("r"))
        return ("sphere", center, float(rad))

    raise TypeError("unsupported target dictionary: %r" % (sorted(d),))


def _normalise_target(target):
    """Return an internal shape descriptor.

    Accepted forms include the classes above, dictionaries (including the
    JSON representation used by ``data/queries.json``), tagged tuples such as
    ``("box", center, half_extents)`` and compound lists of shapes.
    """
    if isinstance(target, Sphere):
        return ("sphere", target.center, target.radius)
    if isinstance(target, Capsule):
        return ("capsule", target.a, target.b, target.radius)
    if isinstance(target, Box):
        return ("box", target.center, target.half_extents, target.rotation)
    if isinstance(target, dict):
        return _shape_from_dict(target)

    if isinstance(target, (list, tuple)):
        if not target:
            raise ValueError("empty target")
        first = target[0]
        if isinstance(first, str):
            tag = first.lower()
            if tag in ("sphere", "ball"):
                return ("sphere", _vec(target[1]), float(target[2]))
            if tag in ("capsule", "segment"):
                return (
                    "capsule",
                    _vec(target[1]),
                    _vec(target[2]),
                    float(target[3]),
                )
            if tag in ("box", "cuboid", "obb"):
                rot = _parse_rotation(target[4]) if len(target) > 4 else None
                return ("box", _vec(target[1]), _vec(target[2]), rot)
            if tag in ("aabb",):
                return _normalise_target({"min": target[1], "max": target[2]})
            if tag in ("compound", "union", "group"):
                return ("compound", tuple(_normalise_target(x) for x in target[1]))
            raise TypeError("unsupported target tag %r" % (first,))
        # A list of shape-like objects is a compound target (union).  This is
        # the representation used by data/queries.json: one target is encoded
        # as ``[{"center": ..., "radius": ...}]``.
        return ("compound", tuple(_normalise_target(x) for x in target))

    raise TypeError("unsupported target type: %s" % type(target).__name__)


def _threshold(shape, radius):
    if shape[0] == "sphere":
        return radius + shape[2]
    if shape[0] == "capsule":
        return radius + shape[3]
    return radius  # box


# ---------------------------------------------------------------------------
# quadratic helpers
# ---------------------------------------------------------------------------


def _solve_quadratic(a, b, c):
    """Solve ``a*t^2 + b*t + c = 0``.

    Returns ``(roots, tangent)``.  Roots are sorted ascending.  A discriminant
    that is zero within a relative tolerance is clamped to zero and reported
    as tangential, which is what makes exact tangency deterministic.
    """
    if a == 0.0:
        if b == 0.0:
            return [], False
        return [-c / b], False

    disc = b * b - 4.0 * a * c
    tol = _EPS * (b * b + abs(4.0 * a * c))
    if disc < -tol:
        return [], False
    tangent = disc <= tol
    if disc < 0.0:
        disc = 0.0
    root = math.sqrt(disc)

    # Stable form: avoids cancellation when |b| is large.
    if b != 0.0:
        q = -0.5 * (b + math.copysign(root, b))
        r1 = q / a
        r2 = c / q if q != 0.0 else r1
    else:
        r1 = -root / (2.0 * a)
        r2 = root / (2.0 * a)
    if r1 > r2:
        r1, r2 = r2, r1
    return [r1, r2], tangent


def _window_time(t, dt):
    """Clamp a root into ``[0, dt]``; ``None`` if it is outside the window."""
    tol = _TIME_EPS * max(1.0, abs(dt))
    if not math.isfinite(t) or t < -tol or t > dt + tol:
        return None
    if t < 0.0:
        return 0.0
    if t > dt:
        return dt
    return t


# ---------------------------------------------------------------------------
# primitive sweeps: moving point against a static shape offset by ``radius``
# ---------------------------------------------------------------------------


def _sweep_point_sphere(p0, v, center, radius, dt):
    """Moving point vs sphere.  Returns ``(t, depth, tangent)`` or None."""
    w = _sub(p0, center)
    ww = _norm2(w)
    rr = radius * radius
    c = ww - rr
    scale = max(1.0, ww, rr)
    if c <= _EPS * scale:  # touching or overlapping at t == 0
        dist = math.sqrt(ww) if ww > 0.0 else 0.0
        touching = abs(ww - rr) <= _EPS * scale
        return 0.0, max(0.0, radius - dist), touching, None

    a = _norm2(v)
    if a == 0.0:
        return None
    b = 2.0 * _dot(w, v)
    roots, tangent = _solve_quadratic(a, b, c)
    for t in roots:
        tc = _window_time(t, dt)
        if tc is not None:
            return tc, 0.0, tangent, None
    return None


def _distance_point_segment(p, a, b):
    d = _sub(b, a)
    dd = _norm2(d)
    if dd <= 0.0:
        return _norm(_sub(p, a)), 0.0
    s = _dot(_sub(p, a), d) / dd
    s = _clamp01(s)
    closest = _add(a, _mul(d, s))
    return _norm(_sub(p, closest)), s


def _sweep_point_capsule(p0, v, a, b, radius, dt):
    """Moving point vs capsule (segment inflated by ``radius``)."""
    d = _sub(b, a)
    dd = _norm2(d)

    if dd == 0.0:  # degenerate capsule is a sphere
        res = _sweep_point_sphere(p0, v, a, radius, dt)
        return res

    # --- already overlapping at t == 0 --------------------------------
    dist0, _ = _distance_point_segment(p0, a, b)
    if dist0 <= radius + _EPS * max(1.0, radius, dist0):
        return 0.0, max(0.0, radius - dist0), False, None

    if _norm2(v) == 0.0:
        return None

    candidates = []

    # --- spherical caps -------------------------------------------------
    res = _sweep_point_sphere(p0, v, a, radius, dt)
    if res is not None:
        candidates.append(res)
    res = _sweep_point_sphere(p0, v, b, radius, dt)
    if res is not None:
        candidates.append(res)

    # --- cylindrical side ----------------------------------------------
    # |(p(t) - a) x d|^2 = radius^2 * |d|^2
    w0 = _sub(p0, a)
    c0 = _cross(w0, d)
    cv = _cross(v, d)
    qa = _norm2(cv)
    if qa != 0.0:
        qb = 2.0 * _dot(c0, cv)
        qc = _norm2(c0) - radius * radius * dd
        roots, tangent = _solve_quadratic(qa, qb, qc)
        s_tol = 1e-9
        for t in roots:
            tc = _window_time(t, dt)
            if tc is None:
                continue
            w = _add(w0, _mul(v, tc))
            s = _dot(w, d) / dd
            if -s_tol <= s <= 1.0 + s_tol:
                candidates.append((tc, 0.0, tangent, None))

    if not candidates:
        return None
    # Earliest candidate; ties prefer a non-tangent candidate.
    candidates.sort(key=lambda item: (item[0], item[2]))
    return candidates[0]


def _dist_point_box_local(q, h):
    dx = abs(q[0]) - h[0]
    dy = abs(q[1]) - h[1]
    dz = abs(q[2]) - h[2]
    if dx < 0.0:
        dx = 0.0
    if dy < 0.0:
        dy = 0.0
    if dz < 0.0:
        dz = 0.0
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _to_local(p, center, rotation):
    w = _sub(p, center)
    if rotation is None:
        return w
    r = rotation
    return (_dot(r[0], w), _dot(r[1], w), _dot(r[2], w))


def _sweep_point_box(p0, v, center, half, rotation, radius, dt):
    """Moving point vs box inflated by ``radius`` (a rounded box)."""
    q0 = _to_local(p0, center, rotation)
    if rotation is None:
        qv = v
    else:
        qv = _to_local(v, (0.0, 0.0, 0.0), rotation)

    h = half
    dist0 = _dist_point_box_local(q0, h)
    if dist0 <= radius + _EPS * max(1.0, radius, dist0):
        return 0.0, max(0.0, radius - dist0), False, None

    if _norm2(qv) == 0.0:
        return None

    # Characteristic length, used only to scale region tests.
    scale = max(
        1.0,
        abs(q0[0]),
        abs(q0[1]),
        abs(q0[2]),
        h[0],
        h[1],
        h[2],
        radius,
        _norm(qv) * dt,
    )
    pos_tol = 1e-9 * scale
    neg_tol = -1e-9 * scale

    candidates = []

    # --- six face patches: |q_i| = h_i + radius -------------------------
    for i in range(3):
        if qv[i] == 0.0:
            continue
        j, k = (i + 1) % 3, (i + 2) % 3
        for sign in (-1.0, 1.0):
            plane = sign * (h[i] + radius)
            t = (plane - q0[i]) / qv[i]
            tc = _window_time(t, dt)
            if tc is None:
                continue
            q = _add(q0, _mul(qv, tc))
            if abs(q[j]) <= h[j] + pos_tol and abs(q[k]) <= h[k] + pos_tol:
                candidates.append((tc, 0.0, False, None))

    # --- twelve edge cylinder patches ----------------------------------
    # For each pair of axes, the four sign combinations of the perpendicular
    # offsets u_i = s_i*q_i - h_i, u_j = s_j*q_j - h_j satisfy
    # u_i^2 + u_j^2 = radius^2, with u_i >= 0, u_j >= 0 and the third
    # coordinate inside its slab.
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        for si in (-1.0, 1.0):
            for sj in (-1.0, 1.0):
                ui0 = si * q0[i] - h[i]
                uj0 = sj * q0[j] - h[j]
                vi = si * qv[i]
                vj = sj * qv[j]
                qa = vi * vi + vj * vj
                if qa == 0.0:
                    continue
                qb = 2.0 * (ui0 * vi + uj0 * vj)
                qc = ui0 * ui0 + uj0 * uj0 - radius * radius
                roots, tangent = _solve_quadratic(qa, qb, qc)
                for t in roots:
                    tc = _window_time(t, dt)
                    if tc is None:
                        continue
                    q = _add(q0, _mul(qv, tc))
                    ui = si * q[i] - h[i]
                    uj = sj * q[j] - h[j]
                    if (
                        ui >= neg_tol
                        and uj >= neg_tol
                        and abs(q[k]) <= h[k] + pos_tol
                    ):
                        candidates.append((tc, 0.0, tangent, None))

    # --- eight corner sphere patches -----------------------------------
    a = _norm2(qv)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = (sx * h[0], sy * h[1], sz * h[2])
                w = _sub(q0, corner)
                qc = _norm2(w) - radius * radius
                qb = 2.0 * _dot(w, qv)
                roots, tangent = _solve_quadratic(a, qb, qc)
                signs = (sx, sy, sz)
                for t in roots:
                    tc = _window_time(t, dt)
                    if tc is None:
                        continue
                    q = _add(q0, _mul(qv, tc))
                    if (
                        signs[0] * q[0] >= h[0] + neg_tol
                        and signs[1] * q[1] >= h[1] + neg_tol
                        and signs[2] * q[2] >= h[2] + neg_tol
                    ):
                        candidates.append((tc, 0.0, tangent, None))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[2]))
    return candidates[0]


def _sweep_point_shape(p0, v, shape, radius, dt):
    kind = shape[0]
    if kind == "sphere":
        return _sweep_point_sphere(p0, v, shape[1], radius + shape[2], dt)
    if kind == "capsule":
        return _sweep_point_capsule(
            p0, v, shape[1], shape[2], radius + shape[3], dt
        )
    if kind == "box":
        return _sweep_point_box(
            p0, v, shape[1], shape[2], shape[3], radius, dt
        )
    if kind == "compound":
        best = None
        for part in shape[1]:
            res = _sweep_point_shape(p0, v, part, radius, dt)
            if res is None:
                continue
            if best is None or (res[0], res[2]) < (best[0], best[2]):
                best = res
        return best
    raise TypeError("unsupported normalised shape %r" % (kind,))


# ---------------------------------------------------------------------------
# public sweep API
# ---------------------------------------------------------------------------


def sweep_sphere(start, velocity, target, dt, radius):
    """First contact time of a moving sphere with one static target.

    Parameters
    ----------
    start : sequence of 3 numbers
        Centre of the moving sphere at ``t = 0``.
    velocity : sequence of 3 numbers
        Constant velocity; centre at time ``t`` is ``start + velocity * t``.
    target : Sphere | Capsule | Box | dict | tagged tuple | compound list
        The static target.  ``data/queries.json`` dictionaries are accepted.
    dt : number
        Length of the time window.  Results are guaranteed to lie in
        ``[0, dt]``.
    radius : number
        Radius of the moving sphere.

    Returns
    -------
    ImpactTime or None
        A float-compatible first-contact time in ``[0, dt]``, or ``None`` when
        there is no contact in the window.  On a hit at ``t == 0`` (the start
        configuration already overlaps), ``.depth`` holds the penetration
        depth.
    """
    dt = float(dt)
    radius = float(radius)
    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be a finite non-negative number")
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be a finite non-negative number")
    start = _vec(start)
    velocity = _vec(velocity)

    shape = _normalise_target(target)
    res = _sweep_point_shape(start, velocity, shape, radius, dt)
    if res is None:
        return None
    t, depth, tangent, _ = res
    return ImpactTime(t, depth=depth, tangent=tangent, target=target)


def _targets_as_list(targets):
    if targets is None:
        return []
    if isinstance(targets, (Sphere, Capsule, Box, dict)):
        return [targets]
    if isinstance(targets, (list, tuple)):
        if not targets:
            return []
        # A tagged compound tuple is one target, not several.
        if isinstance(targets[0], str) and targets[0].lower() in (
            "compound",
            "union",
            "group",
            "sphere",
            "ball",
            "capsule",
            "segment",
            "box",
            "cuboid",
            "obb",
            "aabb",
        ):
            return [targets]
        return list(targets)
    return list(targets)


def time_of_impact(start, velocity, targets, dt, radius):
    """First contact of one moving sphere against multiple static targets.

    Returns a list of ``(ImpactTime, target)`` pairs sorted by ascending time.
    Exactly equal times keep their input order, which makes multi-hit ordering
    stable and reproducible.  Targets with no contact are omitted entirely.

    Batch and per-target queries agree by construction: this function simply
    calls :func:`sweep_sphere` for each target and sorts the successful ones.
    """
    items = _targets_as_list(targets)
    hits = []
    for index, target in enumerate(items):
        t = sweep_sphere(start, velocity, target, dt, radius)
        if t is not None:
            hits.append((t, target, index))
    hits.sort(key=lambda item: (float(item[0]), item[2]))
    return [(t, target) for t, target, _ in hits]
