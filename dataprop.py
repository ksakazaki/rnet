from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict
from heapq import heappop, heappush
from functools import cached_property, partial
import math
import pickle
import time
from typing import Any, Dict, Iterable, List, Set, Union
from yaml import safe_dump
import numpy as np
from rnet.model import Model
from rnet.optimize import Dijkstra, ConnectivityError


class Timer:
    '''
    Timer for timing iterations during an algorithm call.
    '''

    def __init__(self):
        self.exec_times = []

    def __len__(self):
        return len(self.exec_times)

    def clear(self):
        '''
        Clear execution times.
        '''
        self.exec_times.clear()

    def tic(self):
        '''
        Start timer.
        '''
        self.start = time.perf_counter()

    def toc(self):
        '''
        Stop timer and add execution time to the :attr:`exec_times`
        list.
        '''
        self.exec_times.append(time.perf_counter() - self.start)

    @property
    def min(self) -> float:
        return np.min(self.exec_times)

    @property
    def max(self) -> float:
        return np.max(self.exec_times)

    @property
    def avg(self) -> float:
        return np.mean(self.exec_times)

    @property
    def sum(self) -> float:
        return np.sum(self.exec_times)

    def print_stats(self) -> None:
        '''
        Print minimum, maximum, average, and total execution time, as
        well as iteration count.
        '''
        print(f'Recorded {len(self):,} execution times.',
              '\nExec time (sec/iter):',
              f'min {self.min:.04f},',
              f'max {self.max:.04f},',
              f'avg {self.avg:.04f},',
              f'total {self.sum:.04f}')


class DataPropagationProblemSetting:
    '''
    Problem setting for the data propagation problem.

    Parameters
    ----------
    start_node_id, goal_node_id : int
        Start and goal node IDs.
    destination_region_ids : List[int]
        List of region IDs that must be visited.
    min_propagation_time : float
        Minimum propagation time.
    vehicle_speed : float
        Vehicle speed in km/h.
    border_nodes : Dict[int, List[int]]
        Dictionary mapping region ID to list of node IDs that surround
        that region.
    '''

    __slots__ = ['start_node_id', 'goal_node_id', 'destination_region_ids',
                 'num_destinations', 'min_propagation_time', 'vehicle_speed',
                 'search_space_size']

    def __init__(self,
                 start_node_id: int,
                 goal_node_id: int,
                 destination_region_ids: List[int],
                 min_propagation_time: float,
                 vehicle_speed: float,
                 border_nodes: Dict[int, List[int]]):
        self.start_node_id = start_node_id
        self.goal_node_id = goal_node_id
        self.destination_region_ids = destination_region_ids
        self.num_destinations = len(destination_region_ids)
        self.min_propagation_time = min_propagation_time
        self.vehicle_speed = vehicle_speed * 1000 / 3600  # Vehicle speed in m/s
        self.search_space_size = int(
            math.factorial(self.num_destinations) *
            np.prod([len(border_nodes[region])
                     for region in self.destination_region_ids]))

    def __repr__(self):
        return '\n'.join([
            f'Start node: {self.start_node_id}',
            f'Goal node: {self.goal_node_id}',
            f'Destinations: {self.destination_region_ids}',
            f'Propagation time constraint: {self.min_propagation_time} sec',
            f'Vehicle speed: {self.vehicle_speed} m/s',
            f'Search space size: {self.search_space_size:,}',
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}

    def to_yaml(self, output_path: str) -> None:
        '''
        Save genetic algorithm parameters in YAML format.

        Parameters
        ----------
        output_path : str
            Output YAML path.
        '''
        with open(output_path, 'w') as file:
            safe_dump(self.to_dict(), file, indent=2)


@dataclass
class DataPropagationSolver(ABC):
    '''
    Base solver for the data propagation problem.

    Parameters
    ----------
    weights : Dict[int, Dict[int, float]]
        Dictionary whose keys are source nodes and values are a
        mapping from destination nodes to corresponding weights.
    border_nodes : Dict[int, List[int]]
        Dictionary mapping region ID to list of node IDs that surround
        that region.
    area_nodes : Dict[int, Set[int]]
        Dictionary mapping region ID to set of node IDs that are
        contained within that region.

    Attributes
    ----------
    area_weights : Dict[int, Dict[int, Dict[int, float]]]
        ``area_weights[i][j][region_id]`` is the weight of edge
        :math:`(i, j)` if it is contained in the region with the given
        ID. Otherwise, the value is 0.
    '''

    weights: Dict[int, Dict[int, float]]
    border_nodes: Dict[int, List[int]]
    area_nodes: Dict[int, Set[int]]

    def __post_init__(self) -> None:
        self.area_weights = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        for region_id in self.area_nodes.keys():
            area_border_union = \
                self.area_nodes[region_id].union(self.border_nodes[region_id])
            for i in self.weights.keys():
                for j, weight in self.weights[i].items():
                    if i in area_border_union and j in area_border_union:
                        self.area_weights[i][j][region_id] = weight
        for i in self.area_weights.keys():
            for j in self.area_weights[i].keys():
                self.area_weights[i][j] = dict(self.area_weights[i][j])
            self.area_weights[i] = dict(self.area_weights[i])
        self.area_weights = dict(self.area_weights)
        self.shortest_path = Dijkstra(self.weights)
        self.timer = Timer()

    @abstractmethod
    def __call__(self):
        pass

    def get_area_weight(self, i: int, j: int, region_id: int) -> float:
        '''
        Return weight of edge :math:`(i, j)` if it is contained in the
        given region.

        Parameters
        ----------
        i, j : int
            Edge definition.
        region_id : int
            Region ID.

        Returns
        -------
        float
            If edge :math:`(i, j)` is contained in region `region_id`,
            then return edge weight. Otherwise, return 0.0.
        '''
        try:
            return self.area_weights[i][j][region_id]
        except KeyError:
            return 0.0


@dataclass
class DataPropagationBranchAndBound(DataPropagationSolver):
    '''
    Branch and bound solver for the data propagation problem.

    Parameters
    ----------
    weights : Dict[int, Dict[int, float]]
        Dictionary whose keys are source nodes and values are a
        mapping from destination nodes to corresponding weights.
    border_nodes : Dict[int, List[int]]
        Dictionary mapping region ID to list of node IDs that surround
        that region.
    area_nodes : Dict[int, Set[int]]
        Dictionary mapping region ID to set of node IDs that are
        contained within that region.
    node_coords : :class:`~numpy.ndarray`
        Array of node coordinates.
    '''

    node_coords: np.ndarray
    queue_type: str

    def __post_init__(self):
        super().__post_init__()
        self.queue = []
        if self.queue_type == 'fifo':
            self.push = self.queue.append
            self.pop = partial(self.queue.pop, 0)
        elif self.queue_type == 'lifo':
            self.push = self.queue.append
            self.pop = self.queue.pop
        elif self.queue_type == 'priority':
            self.push = partial(heappush, self.queue)
            self.pop = partial(heappop, self.queue)

    def __call__(self, problem_setting: DataPropagationProblemSetting
                 ) -> List[int]:
        '''
        Algorithm call.

        Parameters
        ----------
        problem_setting : :class:`DataPropagationProblemSetting`
            Problem setting for the data propagation problem.

        Returns
        -------
        path : List[int]
            Best solution to the data propagation problem.
        '''
        self.problem_setting = problem_setting
        self.goal_coords = self.node_coords[problem_setting.goal_node_id]

        self.branch_count = 0
        self.prune_count = 0

        self.timer.clear()

        self.timer.tic()
        self.initialize()
        self.timer.toc()

        while self.queue:
            self.timer.tic()
            cost, route, order = self.pop()
            if len(route) == problem_setting.num_destinations:
                self.update(cost, route, order)
            else:
                self.branch(cost, route, order)
            self.timer.toc()
            print(f'Explored {self.branch_count:,} branches,',
                  f'pruned {self.prune_count:,}', end='\r')

        print(f'Explored {self.branch_count:,} branches,',
              f'pruned {self.prune_count:,}')
        self.timer.print_stats()

    def initialize(self) -> None:
        '''
        Initialize queue with all possible first legs.
        '''
        self.best_cost = np.inf
        self.best_route = None
        self.best_order = None
        self.queue.clear()

        start_node_id = self.problem_setting.start_node_id
        destination_region_ids = self.problem_setting.destination_region_ids

        for region_id in destination_region_ids:
            for node_id in self.border_nodes[region_id]:
                try:
                    cost = self.shortest_path(start_node_id, node_id)
                    self.push((cost, [node_id], [region_id]))
                except ConnectivityError:
                    continue

    def heuristic(self, node_id: int) -> float:
        '''
        Return best possible cost from the current node to the goal node.

        Parameters
        ----------
        node_id : int
            Current node ID.

        Returns
        -------
        float
            Straight-line distance from the current node to the goal
            node.
        '''
        return np.linalg.norm(self.node_coords[node_id] - self.goal_coords)

    def branch(self, cost: float, route: List[int], order: List[int]) -> None:
        '''
        Given cost, route, and order, add to queue all possible
        continuations that may have better cost.

        Parameters
        ----------
        cost : float
            Cost so far.
        route : List[int]
            List of waypoints.
        order : List[int]
            List of region IDs.
        '''
        remaining_region_ids = \
            set(self.problem_setting.destination_region_ids).difference(order)
        last_node_id = route[-1]
        for region_id in remaining_region_ids:
            new_order = order + [region_id]
            for node_id in self.border_nodes[region_id]:
                try:
                    new_cost = cost + self.shortest_path(last_node_id, node_id)
                    assert new_cost + self.heuristic(node_id) < self.best_cost
                except (ConnectivityError, AssertionError):
                    self.prune_count += \
                        np.prod([len(self.border_nodes[region])
                                 for region in remaining_region_ids
                                 if region != region_id], dtype=int)
                    continue
                else:
                    new_route = route + [node_id]
                    self.push((new_cost, new_route, new_order))

    def update(self, cost: float, route: List[int], order: List[int]) -> None:
        '''
        If this route and order gives rise to an improved solution, then
        update the :attr:`best_cost`, :attr:`best_route`, and
        :attr:`best_order` attributes.

        Parameters
        ----------
        cost : float
            Cost so far.
        route : List[int]
            List of waypoints.
        order : List[int]
            List of region IDs.
        '''
        self.branch_count += 1
        try:
            final_cost = cost + self.shortest_path(
                route[-1], self.problem_setting.goal_node_id)
        except ConnectivityError:
            return
        if final_cost < self.best_cost:
            times = self.propagation_times(route)
            if not times:
                return
            self.best_cost = final_cost
            self.best_route = route
            self.best_order = order
            self.best_times = times
            print(f'Improved solution: cost={final_cost:.4f},',
                  f'route={route}, order={order}, times={np.round(times, 2)}')

    def propagation_times(self, route: List[int]) -> List[float]:
        '''
        Return propagation times for each destination.

        Parameters
        ----------
        route : List[int]
            List of waypoints.

        Returns
        -------
        List[float]
            Propagation times for each destination.
        '''
        vehicle_speed = self.problem_setting.vehicle_speed
        min_propagation_time = self.problem_setting.min_propagation_time

        # Find total path
        total_route = [self.problem_setting.start_node_id] \
            + route + [self.problem_setting.goal_node_id]
        total_path = [total_route[0]]
        for (start, goal) in zip(total_route[:-1], total_route[1:]):
            total_path += self.shortest_path(start, goal, True)[1]

        # Calculate propagation times to evaluate feasibility
        times = []
        for region_id in self.problem_setting.destination_region_ids:
            dist = 0.0
            for (i, j) in zip(total_path[:-1], total_path[1:]):
                dist += self.get_area_weight(i, j, region_id)
            times.append(dist / vehicle_speed)
            if times[-1] < min_propagation_time:
                return
        return times


@dataclass
class DataPropagationChromosome:
    '''
    Chromosome encoding solution for the data propagation problem.

    Parameters
    ----------
    route : :class:`~numpy.ndarray`, shape (num_dsts,)
        Array of waypoint IDs.
    route_indices : :class:`~numpy.ndarray`, shape (num_dsts,)
        Indices of waypoint IDs.
    order : :class:`~numpy.ndarray`, shape (num_dsts,)
        Indices that sort the waypoint IDs in the order visited by
        this solution.

    Attributes
    ----------
    cost : float
        Total path cost.
    path : List[int]
        Path encoded by this chromosome.
    propagation_times : :class:`~numpy.ndarray`, shape (num_dsts,)
        Propagation times for each destination.
    is_feasible : bool
        Whether all propagation times meet the propagation time
        constriant.
    '''

    route: List[int]
    route_indices: List[int]
    order: List[int]
    cost: float = None
    path: List[int] = None
    propagation_times: np.ndarray = None
    is_feasible: bool = None
    evaluated: bool = False

    @cached_property
    def ordered_route(self) -> List[int]:
        '''
        Array of waypoints in the order visited by this solution.

        Returns
        -------
        List[int]
        '''
        return np.array(self.route)[self.order].tolist()


@dataclass
class DataPropagationPopulation:
    '''
    Class representing a population of chromosomes for the data
    propagation problem.

    Parameters
    ----------
    chromosomes : List[:class:`DataPropgataionChromosome`]
        List of chromosomes.
    '''

    chromosomes: List[DataPropagationChromosome]

    def __getitem__(self, index: Union[int, Iterable[int]]
                    ) -> Union[DataPropagationChromosome, List[DataPropagationChromosome]]:
        if np.issubdtype(type(index), np.integer):
            return self.chromosomes[index]
        else:
            return [self.chromosomes[i] for i in index]

    def __iter__(self):
        return iter(self.chromosomes)

    @cached_property
    def best_chromosome(self) -> DataPropagationChromosome:
        '''
        Feasible chromosome with lowest cost.
        '''
        for index in np.argsort(self.costs):
            if self[index].is_feasible:
                return self[index]

    @cached_property
    def costs(self) -> np.ndarray:
        '''
        Array of chromosome costs.
        '''
        return np.array([chromosome.cost for chromosome in self.chromosomes])

    @cached_property
    def fitness(self) -> np.ndarray:
        '''
        Array of fitness values.
        '''
        return np.array([1/chromosome.cost if chromosome.is_feasible else 0.0
                         for chromosome in self.chromosomes])


@dataclass
class DataPropagationGeneticAlgorithmParams:
    '''
    Parameters for the genetic algorithm.

    Parameters
    ----------
    population_size : int
        Population size.
    crossover_rate : float
        The rate at which parent chromosomes are copied into the child
        population without being recombined.
    selection_size : int
        Selection size for order-based crossover.
    mutation_rate : float
        The rate at which mutations occur.
    regular_mutation_rate : float
        The rate at which regular mutations occur. Of the mutable
        chromosomes, a regular mutation is applied at this rate.
    neighborhood_size : int
        Neighborhood size for local mutations.
    max_iterations : int
        The maximum number of iterations.
    patience : int
        If this many iterations pass without improvement of the best
        solution, then the algorithm is terminated.
    '''

    population_size: int
    crossover_rate: float
    selection_size: int
    mutation_rate: float
    regular_mutation_rate: float
    neighborhood_size: int
    max_iterations: int
    patience: int

    def to_yaml(self, output_path: str) -> None:
        '''
        Save genetic algorithm parameters in YAML format.

        Parameters
        ----------
        output_path : str
            Output YAML path.
        '''
        with open(output_path, 'w') as file:
            safe_dump(asdict(self), file, indent=2)


@dataclass
class DataPropagationGeneticAlgorithm(DataPropagationSolver):
    '''
    Genetic algorithm for the data propagation problem.

    Parameters
    ----------
    weights : Dict[int, Dict[int, float]]
        Dictionary whose keys are source nodes and values are a
        mapping from destination nodes to corresponding weights.
    border_nodes : Dict[int, List[int]]
        Dictionary mapping region ID to list of node IDs that surround
        that region.
    area_nodes : Dict[int, Set[int]]
        Dictionary mapping region ID to set of node IDs that are
        contained within that region.
    params : :class:`DataPropagationGeneticAlgorithmParams`
        Parameters for the genetic algorithm.
    '''

    params: DataPropagationGeneticAlgorithmParams

    def __call__(self, problem_setting: DataPropagationProblemSetting,
                 rng_seed: int = None) -> None:
        '''
        Algorithm call.

        Parameters
        ----------
        problem_setting : :class:`DataPropagationProblemSetting`
            Problem setting for the data propagation problem.
        rng_seed: int or None, optional
            Seed for the random number generator.
        '''
        self.problem_setting = problem_setting
        self.rng = np.random.default_rng(rng_seed)

        self.populations = {}
        self.best_cost = np.inf

        self.timer.clear()

        self.timer.tic()
        self.iter = 0
        self.initialize()
        self.evaluate()
        self.update_best_solution()
        self.timer.toc()

        for i in range(1, self.params.max_iterations + 1):
            self.timer.tic()
            self.iter = i
            self.crossover()
            self.mutate()
            self.evaluate()
            self.update_best_solution()
            self.timer.toc()
            if i - self.best_iteration == self.params.patience:
                break

        self.timer.print_stats()

    def initialize(self) -> None:
        '''
        Generate the initial population.
        '''
        population_size = self.params.population_size
        num_destinations = self.problem_setting.num_destinations
        all_border_nodes = [
            np.array(self.border_nodes[region_id])
            for region_id in self.problem_setting.destination_region_ids]
        route_indices = np.column_stack(
            [self.rng.choice(num_border_nodes, population_size)
             for num_border_nodes in map(len, all_border_nodes)])
        routes = np.column_stack(
            [border_nodes[route_indices[:, col]]
             for col, border_nodes in enumerate(all_border_nodes)])
        orders = np.vstack([self.rng.permutation(num_destinations)
                            for _ in range(population_size)])
        self.populations[0] = DataPropagationPopulation([
            DataPropagationChromosome(list(route), list(indices), list(order))
            for (route, indices, order) in zip(routes, route_indices, orders)])

    def evaluate(self) -> None:
        '''
        Evaluate the current population. Each chromosome is evaluated
        by finding its cost, shortest path, propagation times, and
        feasibility.
        '''
        start_node_id = self.problem_setting.start_node_id
        goal_node_id = self.problem_setting.goal_node_id
        destination_region_ids = self.problem_setting.destination_region_ids
        num_destinations = self.problem_setting.num_destinations
        min_propagation_time = self.problem_setting.min_propagation_time
        for chromosome in self.populations[self.iter]:
            if chromosome.evaluated:
                continue
            ordered_route = \
                [start_node_id] + chromosome.ordered_route + [goal_node_id]
            chromosome.cost = 0.0
            chromosome.path = [start_node_id]
            chromosome.propagation_times = np.zeros(num_destinations)
            for (start, goal) in zip(ordered_route[:-1], ordered_route[1:]):
                try:
                    cost, path = self.shortest_path(start, goal, True)
                except ConnectivityError:
                    chromosome.is_feasible = False
                    break
                chromosome.cost += cost
                chromosome.path += path[1:]
                for (i, j) in zip(path[: -1], path[1:]):
                    chromosome.propagation_times += \
                        np.array([self.get_area_weight(i, j, region_id)
                                  for region_id in destination_region_ids])
            else:
                chromosome.propagation_times /= self.problem_setting.vehicle_speed
                chromosome.is_feasible = \
                    np.all(chromosome.propagation_times > min_propagation_time)
            chromosome.evaluated = True

    def crossover(self) -> None:
        '''
        Select parent chromosomes and perform crossover to generate
        offspring.
        '''
        num_destinations = self.problem_setting.num_destinations
        population_size = self.params.population_size
        selection_size = self.params.selection_size

        parent_population = self.populations[self.iter-1]
        fitness = parent_population.fitness
        probability = fitness / np.sum(fitness)

        children = []
        for crossover in (self.rng.random(population_size//2) < self.params.crossover_rate):
            parents = parent_population[
                self.rng.choice(population_size, 2, False, probability, shuffle=False)]
            if not crossover:
                children.extend([deepcopy(parents[0]), deepcopy(parents[1])])
                continue
            # One-point crossover for routes
            index = self.rng.choice(num_destinations)
            route_A = parents[0].route[:index] + parents[1].route[index:]
            route_B = parents[1].route[:index] + parents[0].route[index:]
            indices_A = parents[0].route_indices[:index] + \
                parents[1].route_indices[index:]
            indices_B = parents[1].route_indices[:index] + \
                parents[0].route_indices[index:]
            # Order-based crossover for orders
            indices = self.rng.choice(
                num_destinations, selection_size, False, shuffle=False)
            order_A = np.array(parents[0].order)
            new_order = np.array(parents[1].order)[indices]
            order_A[np.where(np.isin(order_A, new_order))] = new_order
            order_A = order_A.tolist()
            order_B = np.array(parents[1].order)
            new_order = np.array(parents[0].order)[indices]
            order_B[np.where(np.isin(order_B, new_order))] = new_order
            order_B = order_B.tolist()
            # Create children
            children.extend([DataPropagationChromosome(route_A, indices_A, order_A),
                             DataPropagationChromosome(route_B, indices_B, order_B)])
        self.populations[self.iter] = DataPropagationPopulation(children)

    def mutate(self) -> None:
        '''
        Mutation offspring chromosomes.
        '''
        population = self.populations[self.iter]
        destinations = self.problem_setting.destination_region_ids
        num_destinations = self.problem_setting.num_destinations
        population_size = self.params.population_size
        index_candidates = np.arange(
            -self.params.neighborhood_size, self.params.neighborhood_size + 1)
        mutate = self.rng.random(population_size) < self.params.mutation_rate
        for index in np.flatnonzero(mutate):
            mutate_indices = \
                np.flatnonzero(self.rng.random(num_destinations) < 0.5)
            if self.rng.random() < self.params.regular_mutation_rate:
                # Perform global mutation
                for i in mutate_indices:
                    region_id = destinations[i]
                    num_border_nodes = len(self.border_nodes[region_id])
                    new_node_index = self.rng.choice(num_border_nodes)
                    population[index].route[i] = \
                        self.border_nodes[region_id][new_node_index]
                    population[index].route_indices[i] = \
                        new_node_index
                i, j = self.rng.choice(num_destinations, 2, replace=False)
                population[index].order[i], population[index].order[j] = \
                    population[index].order[j], population[index].order[i]
            else:
                # Perform local mutation
                for i in mutate_indices:
                    region_id = destinations[i]
                    new_node_index = np.mod(
                        population[index].route_indices[i]
                        + self.rng.choice(index_candidates),
                        len(self.border_nodes[region_id]))
                    population[index].route[i] = \
                        self.border_nodes[region_id][new_node_index]
                    population[index].route_indices[i] = \
                        new_node_index
                i = self.rng.choice(num_destinations - 1)
                j = i + 1
                population[index].order[i], population[index].order[j] = \
                    population[index].order[j], population[index].order[i]
            population[index].evaluated = False

    def update_best_solution(self) -> bool:
        '''
        Update best solution. The current iteration and best cost are
        stored in the :attr:`best_iteration` and :attr:`best_cost`
        attributes, respectively.

        Returns
        -------
        improved : bool
            True if the best solution was improved on this iteration,
            otherwise False.
        '''
        best_chromosome = self.populations[self.iter].best_chromosome
        if best_chromosome.cost < self.best_cost:
            self.best_iteration = self.iter
            self.best_cost = best_chromosome.cost
            print(f'Improved solution: cost={best_chromosome.cost:.4f},',
                  f'route={best_chromosome.route},',
                  f'order={best_chromosome.order},',
                  f'times={np.round(best_chromosome.propagation_times, 2)}')
            return True
        return False

    def to_pickle(self, filepath: str) -> None:
        with open(filepath, "wb") as file:
            pickle.dump(self.populations, file)


def run_bb(model: Model, problem_setting: DataPropagationProblemSetting,
           queue_type: str):
    global algorithm
    algorithm = DataPropagationBranchAndBound(
        model.edges.weights(),
        model.border_nodes.to_dict(),
        model.places.area_nodes(model.nodes, 500),
        model.nodes.coords(2),
        queue_type)
    algorithm(problem_setting)


def run_ga(model: Model, solver_params: DataPropagationGeneticAlgorithmParams,
           problem_setting: DataPropagationProblemSetting, output_path: str):
    global algorithm
    algorithm = DataPropagationGeneticAlgorithm(
        model.edges.weights(),
        model.border_nodes.to_dict(),
        model.places.area_nodes(model.nodes, 500),
        solver_params)
    algorithm(problem_setting)
    if output_path:
        algorithm.to_pickle(output_path)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('model_path', help="path to pickled RNet model")
    parser.add_argument('start_node_id', nargs='?', type=int, default=-1)
    parser.add_argument('goal_node_id', nargs='?', type=int, default=-1)
    parser.add_argument('destination_region_ids', nargs='*', type=int)
    parser.add_argument('-bb', '--branch_and_bound', action='store_true')
    parser.add_argument('-fifo', '--breadth_first', action='store_true',
                        help='breadth-first branch and bound search')
    parser.add_argument('-lifo', '--depth_first', action='store_true',
                        help='depth-first branch and bound search')
    parser.add_argument('--num_destinations', type=int)
    parser.add_argument('--min_propagation_time', type=int, default=60)
    parser.add_argument('--vehicle_speed', type=int, default=45)
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--crossover_rate', type=float, default=0.8)
    parser.add_argument('--selection_size', type=int, default=1)
    parser.add_argument('--mutation_rate', type=float, default=0.8)
    parser.add_argument('--regular_mutation_rate', type=float, default=0.5)
    parser.add_argument('--neighborhood_size', type=int, default=2)
    parser.add_argument('--max_iterations', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--output', type=str, default="")
    args = parser.parse_args()

    model = Model.from_pickle(args.model_path)

    start_node_id = args.start_node_id
    if start_node_id == -1:
        start_node_id = np.random.choice(len(model.nodes))

    goal_node_id = args.goal_node_id
    if goal_node_id == -1:
        goal_node_id = np.random.choice(len(model.nodes))

    destination_region_ids = args.destination_region_ids
    if not destination_region_ids:
        assert args.num_destinations, \
            'either destination_region_ids or --num_destinations required'
        destination_region_ids = np.random.choice(len(model.areas),
                                                  args.num_destinations,
                                                  replace=False).tolist()

    problem_setting = DataPropagationProblemSetting(
        start_node_id, goal_node_id, destination_region_ids,
        args.min_propagation_time, args.vehicle_speed,
        model.border_nodes.to_dict())

    print(problem_setting)

    if args.branch_and_bound:
        if args.breadth_first:
            run_bb(model, problem_setting, 'fifo')
        elif args.depth_first:
            run_bb(model, problem_setting, 'lifo')
        else:
            run_bb(model, problem_setting, 'priority')

    else:
        solver_params = DataPropagationGeneticAlgorithmParams(
            args.population_size, args.crossover_rate, args.selection_size,
            args.mutation_rate, args.regular_mutation_rate,
            args.neighborhood_size, args.max_iterations, args.patience)
        run_ga(model, solver_params, problem_setting, args.output)
