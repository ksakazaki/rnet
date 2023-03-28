from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from heapq import heappush, heappop
import time
from typing import Dict, List, Tuple, Union
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
        self.CALL_COUNT += 1
        return result
    return wrapper


class Algorithm(ABC):

    def __init__(self):
        self.EXEC_TIMES = []
        self.CALL_COUNT = 0

    def __init_subclass__(cls) -> None:
        cls.__call__ = run(cls.__call__)
        return super().__init_subclass__()

    @abstractmethod
    def __call__(self):
        pass

    def reset(self):
        self.EXEC_TIMES.clear()
        self.CALL_COUNT = 0


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
    weights : Dict[int, Dict[int, float]]
        Dictionary whose keys are source nodes and values are a
        mapping from destination nodes to corresponding weights.
    '''

    weights: Dict[int, Dict[int, float]]

    def __post_init__(self) -> None:
        self.neighbors = {i: set(self.weights[i]) for i in self.weights.keys()}
        self.visited = {}
        self.queried = {}
        self.origins = {}
        self.queues = {}
        super().__init__()

    def __call__(self, start: int, goal: int, return_path: bool = False
                 ) -> Union[float, Tuple[float, List[int]]]:
        '''
        Algorithm call that returns length of shortest path from
        `start` to `goal`.

        Parameters
        ----------
        start, goal : int
            Start and goal node IDs.
        return_path : bool, optional
            If True, then the shortest path from `start` to `goal` is
            also returned. The default is False.

        Returns
        -------
        cost : float
            Length of shortest path from `start` to `goal`.
        path : List[int], optional
            Path from `start` to `goal`. Only provided if `return_path`
            is True.

        Raises
        ------
        ConnectivityError
            If no path exists from `start` to `goal`.
        '''
        if (start not in self.visited) or (goal not in self.visited[start]):
            self._update(start, goal)
        if return_path:
            origins = self.origins[start]
            path = [goal]
            while path[0] != start:
                path.insert(0, origins[path[0]])
            return self.queried[start][goal], path
        else:
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
                dist = cost_to_node + self.weights[node][neighbor]
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
