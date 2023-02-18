from collections import namedtuple
from typing import Tuple, List
import numpy as np


Circle = namedtuple('Circle', 'x y r')


def circle_intersection(circle1: Circle, circle2: Circle) -> Tuple[List[float]]:
    '''
    Return angles at which two circles intersect each other.
    '''
    dx = circle2.x - circle1.x
    dy = circle2.y - circle1.y
    d = np.linalg.norm([dx, dy])
    t = np.arctan2(dy, dx)
    dt = np.arccos((circle1.r**2 - circle2.r**2 + d**2) / (2 * d * circle1.r))
    return float(np.degrees(t - dt)), float(np.degrees(t + dt))


def polyline_length(coords: np.ndarray) -> float:
    '''
    Return length of a polyline.

    Parameters
    ----------
    coords : :class:`~numpy.ndarray`, shape (N, 2) or (N, 3)
        Coordinates of points along polyline.

    Returns
    -------
    float
    '''
    return float(np.sum(np.linalg.norm(coords[1:] - coords[:-1], axis=1)))
