from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple
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

    def intersects(self, other: 'Circle') -> bool:
        '''
        Return whether this circle intersects another.

        Parameters
        ----------
        other : :class:`Circle`
            Other circle.

        Returns
        -------
        bool
        '''
        dx = other.x - self.x
        dy = other.y - self.y
        return dx ** 2 + dy ** 2 <= (self.r + other.r) ** 2

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
            Intersection angles in radians.
        '''
        if not self.intersects(other):
            return
        dx = other.x - self.x
        dy = other.y - self.y
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
        if not self.intersects(other):
            return
        angles = self.intersection_angles(other)
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

    def points(self, step: float = 0.5) -> np.ndarray:
        '''
        Return perimeter points.

        Parameters
        ----------
        step : float
            Step size in degrees.

        Returns
        -------
        :class:`~numpy.ndarray`
        '''
        t = np.radians(np.arange(0, 360, step))
        return self.center + self.r * np.column_stack((np.cos(t), np.sin(t)))


@dataclass
class Arc:
    '''
    Class for representing an arc.

    Parameters
    ----------
    circle : :class:`Circle`
        Circle instance.
    start, end : float
        Start and end angles in degrees.
    '''

    circle: Circle
    start: float
    end: float

    def __post_init__(self):
        if self.start > self.end:
            self.end += 360

    def points(self, step: float = 0.5):
        '''
        Return arc points.

        Parameters
        ----------
        step : float
            Step size in degrees.

        Returns
        -------
        :class:`~numpy.ndarray`
        '''
        t = np.arange(self.start, self.end, step)
        if t[-1] < self.end:
            t = np.append(t, self.end)
        t = np.radians(t)
        return self.circle.center + self.circle.r * np.column_stack((np.cos(t), np.sin(t)))


def outer_arcs(*circles: Circle) -> List[Arc]:
    '''
    Return outer arcs that form the boundary of the union of `circles`.

    Parameters
    ----------
    *circles : :class:`Circle`
        :class:`Circle` instances.

    Parameters
    ----------
    List[:class:`Arc`]
        List of outer arcs.
    '''
    num_circles = len(circles)
    if num_circles == 1:
        return [Arc(circles[0], 0, 360)]

    # Find angles of intersection between all pairs of circles
    neighbors = defaultdict(set)
    angles = defaultdict(list)
    pairs = {}
    for (i, j) in combinations(range(num_circles), 2):
        if not circles[i].intersects(circles[j]):
            continue
        neighbors[i].add(j)
        neighbors[j].add(i)
        angles[i].extend(circles[i].intersection_angles(circles[j]))
        angles[j].extend(circles[j].intersection_angles(circles[i]))
        index_i, index_j = len(angles[i]) - 1, len(angles[j]) - 1
        pairs[(i, index_i)] = (j, index_j - 1)
        pairs[(j, index_j - 1)] = (i, index_i)
        pairs[(i, index_i - 1)] = (j, index_j)
        pairs[(j, index_j)] = (i, index_i - 1)
    neighbors = dict(neighbors)
    angles = dict(angles)

    # Find outer arcs
    outer_arcs = []
    for circle_id, circle in enumerate(circles):
        sorted_angles = np.sort(angles[circle_id])
        sorted_angles = np.append(sorted_angles, sorted_angles[0] + 2 * np.pi)
        angle_indices = np.argsort(angles[circle_id])
        angle_indices = np.append(angle_indices, angle_indices[0])
        for k in range(len(sorted_angles) - 1):
            theta1, theta2 = sorted_angles[k:k+2]
            alpha = (theta1 + theta2) / 2
            test_point = np.array([circle.x, circle.y]) + \
                circle.r * np.array([np.cos(alpha), np.sin(alpha)])
            for other_id in neighbors[circle_id]:
                if circles[other_id].contains(test_point):
                    break
            else:
                outer_arcs.append((circle_id, *angle_indices[k:k+2]))

    # Sort outer arcs
    sorted_outer_arcs = [outer_arcs.pop()]
    circle_id, start_index, end_index = sorted_outer_arcs[0]
    while outer_arcs:
        circle_id, start_index = pairs[(circle_id, end_index)]
        # Find continuation
        for i, outer_arc in enumerate(outer_arcs):
            if outer_arc[:2] == (circle_id, start_index):
                end_index = outer_arc[2]
                break
        outer_arcs.pop(i)
        sorted_outer_arcs.append((circle_id, start_index, end_index))
    sorted_outer_arcs = [
        Arc(circles[circle_id], np.degrees(angles[circle_id][start_index]),
            np.degrees(angles[circle_id][end_index]))
        for (circle_id, start_index, end_index) in sorted_outer_arcs]
    return sorted_outer_arcs


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
