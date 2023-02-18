from typing import Dict, List, Set


def ccl(neighbors: Dict[int, Set[int]]) -> List[Set[int]]:
    '''
    Clustering via connected-component labeling.

    Parameters
    ----------
    neighbors : Dict[int, Set[int]]
        Dictionary mapping node to its neighbors.

    Returns
    -------
    List[Set[int]]
        List of connected components.
    '''
    clusters = []
    seen = set()
    for s in neighbors:
        if s in seen:
            continue
        clusters.append({s})
        seen.add(s)
        queue = [s]
        while len(queue) > 0:
            m = queue.pop(0)
            for n in neighbors[m]:
                if n in seen:
                    continue
                clusters[-1].add(n)
                seen.add(n)
                queue.append(n)
    return clusters
