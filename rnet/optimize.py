from heapq import heappush, heappop
from typing import Dict, Tuple
import numpy as np
from rnet.utils import neighbors


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


class Dijkstra:

    def __new__(cls, weights: Dict[Tuple[int, int], float]):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Dijkstra, cls).__new__(cls)
        return cls._instance

    def __init__(self, weights: Dict[Tuple[int, int], float]):
        self._weights = weights
        self._neighbors = neighbors(np.array(list(weights)), ordered=False)
        self._queues = {}
        self.visited = {}
        self.queried = {}
        self.origins = {}

    def __call__(self, s: int, g: int):
        try:
            return self.queried[s][g]
        except KeyError:
            self._update(s, g)
            return self.queried[s][g]

    def _update(self, s: int, g: int):
        visited = self.visited.setdefault(s, set())
        queried = self.queried.setdefault(s, {})
        origins = self.origins.setdefault(s, {})
        queue = self._queues.setdefault(s, [])
        if len(visited) == 0:
            heappush(queue, (0.0, s))
        while queue:
            c, n = heappop(queue)
            if n == g:
                heappush(queue, (c, n))
                break
            for m in self.neighbors[n].difference(visited):
                d = c + self._weights[(n, m)]
                if d < queried.get(m, np.inf):
                    queried[m] = d
                    origins[m] = n
                    heappush(queue, (d, m))
            visited.add(n)
        else:
            raise ConnectivityError(s, g)

    def clear(self):
        self.visited.clear()
        self.queried.clear()
        self.origins.clear()
        self.queues.clear()
