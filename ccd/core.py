# -*- coding: utf-8 -*-
"""ccd"""

from __future__ import annotations

__all__ = ["Capsule", "Sphere", "sweep_sphere", "time_of_impact"]


class Sphere:
    """Sphere"""

    def __init__(self, center, radius):
        """球"""
        raise NotImplementedError("Sphere.__init__ 尚未实现")

class Capsule:
    """Capsule"""

    def __init__(self, a, b, radius):
        """胶囊"""
        raise NotImplementedError("Capsule.__init__ 尚未实现")


def sweep_sphere(start, velocity, target, dt, radius):
    """返回首次接触时间，无接触返回 None"""
    raise NotImplementedError("sweep_sphere 尚未实现")


def time_of_impact(start, velocity, targets, dt, radius):
    """多目标求交，返回 [(时间, 目标)]"""
    raise NotImplementedError("time_of_impact 尚未实现")
