from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from heapq import heappush, heappop
import time
from typing import Dict, Tuple
import numpy as np


def run(func):
    '''
    Wrapper for algorithm calls.
    '''
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        self.EXEC_TIMES.append(end_time - start_time)
        return result
    return wrapper


class Algorithm(ABC):

    def __init__(self):
        self.EXEC_TIMES = []

    def __init_subclass__(cls) -> None:
        cls.__call__ = run(cls.__call__)
        return super().__init_subclass__()

    @abstractmethod
    def __call__(self):
        pass

    def clear(self):
        self.EXEC_TIMES.clear()


class Error(Exception):
    pass


class ConnectivityError(Exception):
    '''
    Raised if a path between nodes does not exist.

    Parameters
    ----------
    s, g : int
        Source and destination node IDs.
    '''

    def __init__(self, s: int, g: int) -> None:
        super().__init__(f'no path from {s} to {g}')


@dataclass
class Dijkstra(Algorithm):
    '''
    Implementation of Dijkstra's shortest path algorithm.

    Parameters
    ----------
    weights : Dict[Tuple[int, int], float]
        Dictionary mapping directed connection :math:`(i, j)` to
        corresponding weight.
    '''

    weights: Dict[Tuple[int, int], float]

    def __post_init__(self) -> None:
        neighbors = defaultdict(set)
        for (i, j) in self.weights.keys():
            neighbors[i].add(j)
        self.neighbors = dict(neighbors)
        self.visited = {}
        self.queried = {}
        self.origins = {}
        self.queues = {}
        super().__init__()

    def __call__(self, start: int, goal: int) -> float:
        '''
        Algorithm call that returns length of shortest path from
        `start` to `goal`.

        Parameters
        ----------
        start, goal : int
            Start and goal node IDs.

        Returns
        -------
        float
            Length of shortest path from `start` to `goal`.

        Raises
        ------
        ConnectivityError
            If no path exists from `start` to `goal`.
        '''
        try:
            return self.queried[start][goal]
        except KeyError:
            self._update(start, goal)
            return self.queried[start][goal]

    def _update(self, start: int, goal: int) -> None:
        visited = self.visited.setdefault(start, set())
        queried = self.queried.setdefault(start, {})
        origins = self.origins.setdefault(start, {})
        queue = self.queues.setdefault(start, [])
        if len(visited) == 0:
            heappush(queue, (0.0, start))
        while queue:
            cost_to_node, node = heappop(queue)
            if node == goal:
                heappush(queue, (cost_to_node, node))
                break
            for neighbor in self.neighbors[node].difference(visited):
                dist = cost_to_node + self.weights[(node, neighbor)]
                if dist < queried.get(neighbor, np.inf):
                    queried[neighbor] = dist
                    origins[neighbor] = node
                    heappush(queue, (dist, neighbor))
            visited.add(node)
        else:
            raise ConnectivityError(start, goal)

    def clear(self) -> None:
        self.visited.clear()
        self.queried.clear()
        self.origins.clear()
        self.queues.clear()
        super().clear()
