from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class Circle:
    '''
    Class representing a circle.

    Parameters
    ----------
    x, y, r : float
        Circle center and radius.
    '''

    x: float
    y: float
    r: float

    @property
    def center(self) -> np.ndarray:
        '''
        Circle center.

        Returns
        -------
        :class:`~numpy.ndarray`
        '''
        return np.array([self.x, self.y])

    def intersection_angles(self, other: 'Circle') -> Tuple[float, float]:
        '''
        Return angles at which this circle intersects another.

        Parameters
        ----------
        other : :class:`Circle`
            Other circle.

        Returns
        -------
        Tuple[float, float]
        '''
        dx = other.x - self.x
        dy = other.y - self.y
        if dx ** 2 + dy ** 2 > (self.r + other.r) ** 2:
            # no intersection
            return
        d = np.linalg.norm([dx, dy])
        t = np.arctan2(dy, dx)
        dt = np.arccos((self.r**2 - other.r**2 + d**2) / (2 * d * self.r))
        return (t - dt, t + dt)

    def intersection_points(self, other: 'Circle') -> np.ndarray:
        '''
        Returns points of intersection with another circle.

        Parameters
        ----------
        other : :class:`Circle`
            Other circle.

        Returns
        -------
        :class:`~numpy.ndarray`
        '''
        angles = self.intersection_angles(other)
        if angles is None:
            return
        t1, t2 = angles
        C1, C2 = np.cos([t1, t2])
        S1, S2 = np.sin([t1, t2])
        return np.array([self.x, self.y]) + self.r * np.array([[C1, S1], [C2, S2]])

    def contains(self, point: np.ndarray) -> bool:
        '''
        Return True if circle contains `point`.

        Returns
        -------
        bool
        '''
        return np.sum(np.power(self.center - point, 2)) <= self.r ** 2

    def points(self, start: float = 0.0, end: float = 2*np.pi,
               step: float = 1e-2) -> np.ndarray:
        '''
        Return points on arc between given angles.

        Parameters
        ----------
        start, end : float
            Start and end angles in radians.

        Returns
        -------
        :class:`~numpy.ndarray`
        '''
        t = np.arange(start, end + step, step)
        return self.center + self.r * np.column_stack((np.cos(t), np.sin(t)))


def outer_arcs(circles: Dict[int, Circle]) -> Dict[int, List[Tuple[float, float]]]:
    '''
    Parameters
    ----------
    circles : Dict[int, :class:`Circle`]
        Dictionary mapping ID to :class:`Circle` instance.

    Parameters
    ----------
    Dict[int, List[Tuple[float, float]]]
        Dictionary mapping ID to corresponding outer arcs.
    '''
    # Find angles of intersection between all pairs of circles
    pairs = defaultdict(set)
    angles = defaultdict(list)
    for (i, j) in permutations(list(circles), 2):
        intersection_angles = circles[i].intersection_angles(circles[j])
        if intersection_angles is not None:
            pairs[i].add(j)
            angles[i].extend(intersection_angles)
    pairs = dict(pairs)
    angles = dict(angles)

    # Find outer arcs
    outer_arcs = defaultdict(list)
    for self_id in angles.keys():
        self = circles[self_id]
        sorted_angles = list(sorted(angles[self_id]))
        sorted_angles += [sorted_angles[0] + 360]
        num_arcs = len(sorted_angles) - 1
        for k in range(num_arcs):
            theta1, theta2 = sorted_angles[k:k+2]
            alpha = (theta1 + theta2) / 2
            test_point = np.array([self.x, self.y]) + \
                self.r * np.array([np.cos(alpha), np.sin(alpha)])
            for other_id in pairs[self_id]:
                if circles[other_id].contains(test_point):
                    break
            else:
                outer_arcs[self_id].append((theta1, theta2))
    return outer_arcs


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
