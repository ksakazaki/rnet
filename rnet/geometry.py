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

    def intersections(self, link_coords: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Yield points of intersection between circle and links.
        Intersection points are ordered in counter-clockwise direction
        around the circle.

        Parameters
        ----------
        link_coords : :class:`~numpy.ndarray`
            Two-dimensional link coordinates.

        Returns
        -------
        indices : :class:`~numpy.ndarray`
            Link indices.
        angles : :class:`~numpy.ndarray`
            Intersection angles in degrees.
        points : :class:`~numpy.ndarray`
            Two-dimensional intersection points.

        References
        ----------
        https://mathworld.wolfram.com/Circle-LineIntersection.html
        '''
        dx = np.diff(link_coords[:, :, 0], axis=1).flatten()
        dy = np.diff(link_coords[:, :, 1], axis=1).flatten()
        dr_sq = dx ** 2 + dy ** 2
        D = link_coords[:, 0, 0] * link_coords[:, 1, 1] - \
            link_coords[:, 1, 0] * link_coords[:, 0, 1]

        discriminant = self.r ** 2 * dr_sq - D ** 2
        discriminant_sqrt = np.sqrt(discriminant)
        mask = discriminant > 0
        indices = np.flatnonzero(mask)

        dx = dx[mask]
        dy = dy[mask]
        dr_sq = dr_sq[mask]
        D = D[mask]
        discriminant_sqrt = discriminant_sqrt[mask]

        x = np.hstack((
            D * dy + np.sign(dy) * dx * discriminant_sqrt,
            D * dy - np.sign(dy) * dx * discriminant_sqrt
        )) / np.hstack((dr_sq, dr_sq))
        y = np.hstack((
            -D * dx + np.abs(dy) * discriminant_sqrt,
            -D * dx - np.abs(dy) * discriminant_sqrt
        )) / np.hstack((dr_sq, dr_sq))
        indices = np.hstack((indices, indices))

        scale_factors = (x / np.hstack((dx, dx)))
        mask = (0 < scale_factors) & (scale_factors < 1)
        x = x[mask]
        y = y[mask]
        indices = indices[mask]

        angles = np.degrees(np.mod(np.arctan2(y, x), 2 * np.pi))
        sorted_indices = np.argsort(angles)
        indices = indices[sorted_indices]
        angles = angles[sorted_indices]
        x = x[sorted_indices]
        y = y[sorted_indices]
        points = self.center + np.column_stack((x, y))
        return indices, angles, points

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

    def contains(self, angles: np.ndarray) -> np.ndarray:
        '''
        Return boolean array that is True where angle is inside arc,
        and False otherwise.

        Parameters
        ----------
        angles : :class:`~numpy.ndarray`
            Array of angles in degrees.

        Returns
        -------
        :class:`~numpy.ndarray`
            Boolean array that is True where angle is inside arc, and
            False otherwise.
        '''
        return np.any((
            (self.start < angles) & (angles < self.end),
            (self.start < angles + 360) & (angles + 360 < self.end)), axis=0)

    def intersections(self, link_coords: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Yield points of intersection between arc and links.
        Intersection points are ordered in counter-clockwise direction
        along the arc.

        Parameters
        ----------
        link_coords : :class:`~numpy.ndarray`
            Two-dimensional link coordinates.

        Returns
        -------
        indices : :class:`~numpy.ndarray`
            Link indices.
        angles : :class:`~numpy.ndarray`
            Intersection angles in degrees.
        points : :class:`~numpy.ndarray`
            Two-dimensional intersection points.

        See also
        --------
        :meth:`Circle.intersections`
            Yield points of intersection between circle and links.
        '''
        indices, angles, points = self.circle.intersections(link_coords)
        mask = self.contains(angles)
        indices = indices[mask]
        angles = angles[mask]
        points = points[mask]
        return indices, angles, points

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
