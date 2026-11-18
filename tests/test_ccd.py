# -*- coding: utf-8 -*-
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ccd as mod  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def test_head_on_analytic():
    t = mod.sweep_sphere([0, 0, 0], [1, 0, 0], mod.Sphere([3, 0, 0], 1), 5.0, 0.5)
    assert t is not None and abs(t - 1.5) < 0.05


def test_already_overlapping_returns_zero():
    t = mod.sweep_sphere([0, 0, 0], [1, 0, 0], mod.Sphere([0.5, 0, 0], 1), 5.0, 0.5)
    assert t == 0.0


def test_no_hit_returns_none():
    t = mod.sweep_sphere([0, 0, 0], [-1, 0, 0], mod.Sphere([3, 0, 0], 1), 5.0, 0.5)
    assert t is None


def test_zero_velocity():
    t = mod.sweep_sphere([0, 0, 0], [0, 0, 0], mod.Sphere([5, 0, 0], 1), 5.0, 0.5)
    assert t is None


def test_multi_sorted_ascending():
    targets = [mod.Sphere([5, 0, 0], 1), mod.Sphere([2, 0, 0], 1), mod.Sphere([9, 0, 0], 1)]
    hits = mod.time_of_impact([0, 0, 0], [1, 0, 0], targets, 10.0, 0.5)
    times = [h[0] for h in hits]
    assert times == sorted(times)


def test_tangent_deterministic():
    a = mod.sweep_sphere([0, 0, 0], [1, 0, 0], mod.Sphere([5, 2, 0], 1), 10.0, 1.0)
    b = mod.sweep_sphere([0, 0, 0], [1, 0, 0], mod.Sphere([5, 2, 0], 1), 10.0, 1.0)
    assert a == b
