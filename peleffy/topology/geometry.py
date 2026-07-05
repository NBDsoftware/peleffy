"""
This module contains standard internal-coordinate geometry helpers:
computing bond lengths, bond angles and dihedral angles from Cartesian
coordinates, and the inverse operation of placing a new point from a
set of internal coordinates and three reference points (a NeRF-style
placement).

These use the standard textbook conventions (bond angle with the
vertex at the second point, right-handed dihedral), independent of
the convention used by peleffy.topology.zmatrix.ZMatrix, which follows
a different, PlopRotTemp-inspired definition.
"""

import math


def _subtract(p1, p2):
    return (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])


def _add(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2])


def _dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def _cross(v1, v2):
    return (v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0])


def _norm(v):
    return math.sqrt(_dot(v, v))


def _scale(v, factor):
    return (v[0] * factor, v[1] * factor, v[2] * factor)


def _normalize(v):
    return _scale(v, 1.0 / _norm(v))


def arbitrary_perpendicular(v):
    """
    It returns an arbitrary but deterministic unit vector that is
    perpendicular to v. It is used as a fallback reference direction
    when there are not enough real reference points available to
    define a dihedral angle unambiguously.

    Parameters
    ----------
    v : tuple[float, float, float]
        The vector to which the returned vector will be perpendicular

    Returns
    -------
    perpendicular : tuple[float, float, float]
        A unit vector perpendicular to v
    """
    v_hat = _normalize(v)
    reference = (0.0, 0.0, 1.0)
    if abs(_dot(v_hat, reference)) > 0.9:
        reference = (1.0, 0.0, 0.0)
    return _normalize(_cross(v_hat, reference))


def calculate_bond_length(p1, p2):
    """
    It calculates the distance between two points.

    Parameters
    ----------
    p1 : tuple[float, float, float]
        The xyz coordinates of the first point
    p2 : tuple[float, float, float]
        The xyz coordinates of the second point

    Returns
    -------
    bond_length : float
        The distance between p1 and p2
    """
    return _norm(_subtract(p1, p2))


def calculate_bond_angle(p1, p2, p3):
    """
    It calculates the standard bond angle p1-p2-p3, with the vertex
    at p2, in degrees.

    Parameters
    ----------
    p1 : tuple[float, float, float]
        The xyz coordinates of the first point
    p2 : tuple[float, float, float]
        The xyz coordinates of the second point (the vertex)
    p3 : tuple[float, float, float]
        The xyz coordinates of the third point

    Returns
    -------
    bond_angle : float
        The angle, in degrees, contained between 0 and 180
    """
    v1 = _subtract(p1, p2)
    v2 = _subtract(p3, p2)
    cos_angle = _dot(v1, v2) / (_norm(v1) * _norm(v2))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def calculate_dihedral_angle(p1, p2, p3, p4):
    """
    It calculates the standard dihedral angle p1-p2-p3-p4, i.e. the
    rotation around the p2-p3 axis, in degrees.

    Parameters
    ----------
    p1 : tuple[float, float, float]
        The xyz coordinates of the first point
    p2 : tuple[float, float, float]
        The xyz coordinates of the second point
    p3 : tuple[float, float, float]
        The xyz coordinates of the third point
    p4 : tuple[float, float, float]
        The xyz coordinates of the fourth point

    Returns
    -------
    dihedral_angle : float
        The dihedral angle, in degrees, contained between -180 and 180
    """
    b1 = _subtract(p2, p1)
    b2 = _subtract(p3, p2)
    b3 = _subtract(p4, p3)

    n1 = _cross(b1, b2)
    n2 = _cross(b2, b3)
    m1 = _cross(n1, _normalize(b2))

    return math.degrees(math.atan2(_dot(m1, n2), _dot(n1, n2)))


def place_atom(parent, grandparent, great_grandparent,
               bond_length, bond_angle, dihedral_angle):
    """
    It places a new point at a given bond length, bond angle and
    dihedral angle with respect to three reference points. It is the
    inverse operation of calculate_bond_length, calculate_bond_angle
    and calculate_dihedral_angle altogether (a NeRF-style placement).

    Parameters
    ----------
    parent : tuple[float, float, float]
        The xyz coordinates of the parent point (the one the new
        point will be bonded to)
    grandparent : tuple[float, float, float]
        The xyz coordinates of the grandparent point
    great_grandparent : tuple[float, float, float]
        The xyz coordinates of the great-grandparent point
    bond_length : float
        The target distance between the new point and parent
    bond_angle : float
        The target angle, in degrees, between the new point, parent
        and grandparent (vertex at parent)
    dihedral_angle : float
        The target dihedral angle, in degrees, between the new point,
        parent, grandparent and great-grandparent

    Returns
    -------
    new_point : tuple[float, float, float]
        The resulting xyz coordinates of the new point
    """
    theta = math.radians(bond_angle)
    phi = math.radians(dihedral_angle)

    e_x = _normalize(_subtract(grandparent, parent))

    towards_ggparent = _subtract(great_grandparent, grandparent)
    in_plane = _subtract(towards_ggparent,
                         _scale(e_x, _dot(towards_ggparent, e_x)))

    # Fall back to an arbitrary perpendicular direction if parent,
    # grandparent and great-grandparent happen to be collinear
    if _norm(in_plane) < 1e-6:
        e_y = arbitrary_perpendicular(e_x)
    else:
        e_y = _normalize(in_plane)

    e_z = _cross(e_x, e_y)

    direction = _add(
        _add(_scale(e_x, math.cos(theta)),
             _scale(e_y, math.sin(theta) * math.cos(phi))),
        _scale(e_z, math.sin(theta) * math.sin(phi)))

    return _add(parent, _scale(direction, bond_length))
