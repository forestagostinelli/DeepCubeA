from typing import List, Tuple, Dict, Callable, Optional, Any
from environments.environment_abstract import Environment, State
import numpy as np
from heapq import heappush, heappop

from argparse import ArgumentParser
import torch
from utils import env_utils, nnet_utils, search_utils, misc_utils
import pickle
import time


class Node:
    __slots__ = ['state', 'path_cost', 'heuristic', 'cost', 'is_solved', 'parent_move', 'parent',
                 'transition_costs', 'children', 'backup', 'hash_rep']

    def __init__(self, state: State, path_cost: float, heuristic: float, cost: float, is_solved: bool,
                 parent_move: Optional[int], parent, hash_rep: str):
        self.state: State = state
        self.path_cost: float = path_cost
        self.heuristic: float = heuristic
        self.cost: float = cost
        self.is_solved: bool = is_solved
        self.parent_move: Optional[int] = parent_move
        self.parent: Optional[Node] = parent
        self.transition_costs: List[float] = []
        self.children: List[Node] = []
        self.backup: float = np.inf
        self.hash_rep: str = hash_rep


OpenSetElem = Tuple[float, int, Node]


class Instance:

    def __init__(self, state: State, heuristic: float, cost: float, is_solved, str_rep: str):
        self.open_set: List[OpenSetElem] = []
        self.heappush_count: int = 0
        self.closed_dict: Dict[str, Node] = dict()
        self.popped_nodes: List[Node] = []
        self.goal_nodes: List[Node] = []
        self.num_nodes_generated: int = 0

        self.root_node: Node = Node(state, 0.0, heuristic, cost, is_solved, None, None, str_rep)

        self.push_to_open(self.root_node)

    def push_to_open(self, node: Node):
        heappush(self.open_set, (node.cost, self.heappush_count, node))
        self.heappush_count += 1
        self.closed_dict[node.hash_rep] = node

    def pop_from_open(self, num_nodes: int) -> List[Node]:
        num_to_pop: int = min(num_nodes, len(self.open_set))

        popped_nodes = [heappop(self.open_set)[2] for _ in range(num_to_pop)]
        self.goal_nodes.extend([node for node in popped_nodes if node.is_solved])
        self.popped_nodes.extend(popped_nodes)

        return popped_nodes

    def remove_in_closed(self, nodes: List[Node]) -> List[Node]:
        nodes_not_in_closed: List[Node] = []

        for node in nodes:
            node_get: Optional[Node] = self.closed_dict.get(node.hash_rep)
            if node_get is None:
                nodes_not_in_closed.append(node)
            elif node_get.path_cost > node.path_cost:
                nodes_not_in_closed.append(node)

        return nodes_not_in_closed


def pop_from_open(instances: List[Instance], batch_size: int) -> List[List[Node]]:
    popped_nodes_all: List[List[Node]] = [instance.pop_from_open(batch_size) for instance in instances]

    return popped_nodes_all


def expand_nodes(instances: List[Instance], popped_nodes_all: List[List[Node]],
                 env: Environment) -> List[List[Node]]:
    # Get children of all nodes at once (for speed)
    popped_nodes_flat: List[Node]
    split_idxs: List[int]
    popped_nodes_flat, split_idxs = misc_utils.flatten_list(popped_nodes_all)

    if len(popped_nodes_flat) == 0:
        return [[]]

    states: List[State] = [x.state for x in popped_nodes_flat]

    states_c_by_node: List[List[State]]
    tcs_np: List[np.ndarray]
    states_c_by_node, tcs_np = env.expand(states)
    tcs_by_node: List[List[float]] = [list(x) for x in tcs_np]

    # Get is_solved on all states at once (for speed)
    states_c: List[State]
    states_c, split_idxs_c = misc_utils.flatten_list(states_c_by_node)
    is_solved_c: List[bool] = list(env.is_solved(states_c))
    is_solved_c_by_node: List[List[bool]] = misc_utils.flat_to_list_of_lists(is_solved_c, split_idxs_c)

    # Update path costs for all states at once (for speed)
    parent_path_costs = np.expand_dims(np.array([node.path_cost for node in popped_nodes_flat]), 1)
    path_costs_c: List[float] = (parent_path_costs + np.array(tcs_by_node)).flatten().tolist()
    for node, tcs_c in zip(popped_nodes_flat, tcs_by_node):
        node.transition_costs.extend(tcs_c)

    path_costs_c_by_node: List[List[float]] = misc_utils.flat_to_list_of_lists(path_costs_c, split_idxs_c)

    # Reshape lists
    patch_costs_c_by_inst_node: List[List[List[float]]] = misc_utils.flat_to_list_of_lists(path_costs_c_by_node,
                                                                                           split_idxs)
    states_c_by_inst_node: List[List[List[State]]] = misc_utils.flat_to_list_of_lists(states_c_by_node, split_idxs)
    is_solved_c_by_inst_node: List[List[List[bool]]] = misc_utils.flat_to_list_of_lists(is_solved_c_by_node, split_idxs)

    # Get child nodes
    instance: Instance
    nodes_c_by_inst: List[List[Node]] = []
    for inst_idx, instance in enumerate(instances):
        nodes_c_by_inst.append([])
        parent_nodes: List[Node] = popped_nodes_all[inst_idx]
        path_costs_c_by_node: List[List[float]] = patch_costs_c_by_inst_node[inst_idx]
        states_c_by_node: List[List[State]] = states_c_by_inst_node[inst_idx]
        is_solved_c_by_node: List[List[bool]] = is_solved_c_by_inst_node[inst_idx]

        parent_node: Node
        tcs_node: List[float]
        states_c: List[State]
        for parent_node, path_costs_c, states_c, is_solved_c in zip(parent_nodes, path_costs_c_by_node,
                                                                    states_c_by_node, is_solved_c_by_node):
            str_reps: List[str] = env.get_str_rep(states_c)

            state: State
            for move_idx, state in enumerate(states_c):
                path_cost: float = path_costs_c[move_idx]
                is_solved: bool = is_solved_c[move_idx]
                str_rep: str = str_reps[move_idx]
                node_c: Node = Node(state, path_cost, 0.0, 0.0, is_solved, move_idx, parent_node, str_rep)

                nodes_c_by_inst[inst_idx].append(node_c)

                parent_node.children.append(node_c)

        instance.num_nodes_generated += len(nodes_c_by_inst[inst_idx])

    return nodes_c_by_inst


def remove_in_closed(instances: List[Instance], nodes_c_all: List[List[Node]]) -> List[List[Node]]:
    for inst_idx, instance in enumerate(instances):
        nodes_c_all[inst_idx] = instance.remove_in_closed(nodes_c_all[inst_idx])

    return nodes_c_all


def add_heuristic_and_cost(nodes: List[List[Node]], heuristic_fn: Callable, weight: float) -> None:
    # flatten nodes
    nodes_flat: List[Node]
    nodes_flat, _ = misc_utils.flatten_list(nodes)

    if len(nodes_flat) == 0:
        return

    # get heuristic
    states_flat: List[State] = [node.state for node in nodes_flat]

    # compute node cost
    heuristics_flat = heuristic_fn(states_flat)
    path_costs_flat: np.ndarray = np.array([node.path_cost for node in nodes_flat])
    is_solved_flat: np.ndarray = np.array([node.is_solved for node in nodes_flat])

    costs_flat: np.ndarray = weight * path_costs_flat + heuristics_flat * np.logical_not(is_solved_flat)

    # add heuristic and cost to node
    for node, heuristic, cost in zip(nodes_flat, heuristics_flat, costs_flat):
        node.heuristic = heuristic
        node.cost = cost


def add_to_open(instances: List[Instance], nodes: List[List[Node]]) -> None:
    nodes_inst: List[Node]
    instance: Instance
    for instance, nodes_inst in zip(instances, nodes):
        node: Node
        for node in nodes_inst:
            instance.push_to_open(node)


def get_path(node: Node) -> Tuple[List[State], List[int], float]:
    path: List[State] = []
    moves: List[int] = []

    parent_node: Node = node
    while parent_node.parent is not None:
        path.append(parent_node.state)

        moves.append(parent_node.parent_move)
        parent_node = parent_node.parent

    path.append(parent_node.state)

    path = path[::-1]
    moves = moves[::-1]

    return path, moves, node.path_cost


class AStar:

    def __init__(self, states: List[State], env: Environment, heuristic_fn: Callable, weight: float = 1.0):
        self.env: Environment = env
        self.weight: float = weight
        self.step_num: int = 0

        self.timings: Dict[str, float] = {"pop": 0.0, "expand": 0.0, "check": 0.0, "heur": 0.0,
                                          "add": 0.0, "itr": 0.0}

        # compute starting costs
        heuristics: np.ndarray = heuristic_fn(states)
        is_solved_states: np.ndarray = self.env.is_solved(states)
        costs: np.ndarray = heuristics*np.logical_not(is_solved_states)

        # initialize instances
        self.instances: List[Instance] = []

        str_reps: List[str] = self.env.get_str_rep(states)
        state: State
        for state, heuristic, cost, is_solved_state, str_rep in zip(states, heuristics, costs, is_solved_states,
                                                                    str_reps):
            self.instances.append(Instance(state, heuristic, cost, is_solved_state, str_rep))

    def step(self, heuristic_fn: Callable, batch_size: int, include_solved: bool = False, verbose: bool = False):
        start_time_itr = time.time()
        instances: List[Instance]
        if include_solved:
            instances = self.instances
        else:
            instances = [instance for instance in self.instances if len(instance.goal_nodes) == 0]

        # Pop from open
        start_time = time.time()
        popped_nodes_all: List[List[Node]] = pop_from_open(instances, batch_size)
        pop_time = time.time() - start_time

        # Expand nodes
        start_time = time.time()
        nodes_c_all: List[List[Node]] = expand_nodes(instances, popped_nodes_all, self.env)
        expand_time = time.time() - start_time

        # Get heuristic of children
        start_time = time.time()
        add_heuristic_and_cost(nodes_c_all, heuristic_fn, self.weight)
        heur_time = time.time() - start_time

        # Check if children are in closed
        start_time = time.time()
        nodes_c_all = remove_in_closed(instances, nodes_c_all)
        check_time = time.time() - start_time

        # Add to open
        start_time = time.time()
        add_to_open(instances, nodes_c_all)
        add_time = time.time() - start_time

        itr_time = time.time() - start_time_itr
        # Print to screen
        if verbose:
            print("Itr: %i, Times - pop: %.2f, expand: %.2f, heur: %.2f, check: %.2f, "
                  "add: %.2f, itr: %.2f" % (self.step_num, pop_time, expand_time, heur_time, check_time, add_time,
                                            itr_time))

        # Update timings
        self.timings['pop'] += pop_time
        self.timings['expand'] += expand_time
        self.timings['heur'] += heur_time
        self.timings['check'] += check_time
        self.timings['add'] += add_time
        self.timings['itr'] += itr_time

        self.step_num += 1

    def has_found_goal(self) -> List[bool]:
        goal_found: List[bool] = [len(self.get_goal_nodes(idx)) > 0 for idx in range(len(self.instances))]

        return goal_found

    def get_goal_nodes(self, inst_idx) -> List[Node]:
        return self.instances[inst_idx].goal_nodes

    def get_goal_node_smallest_path_cost(self, inst_idx) -> Node:
        goal_nodes: List[Node] = self.get_goal_nodes(inst_idx)
        path_costs: List[float] = [node.path_cost for node in goal_nodes]

        goal_node: Node = goal_nodes[int(np.argmin(path_costs))]

        return goal_node

    def get_num_nodes_generated(self, inst_idx: int) -> int:
        return self.instances[inst_idx].num_nodes_generated

    def get_popped_nodes(self) -> List[List[Node]]:
        popped_nodes_all: List[List[Node]] = [instance.popped_nodes for instance in self.instances]
        return popped_nodes_all


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--states', type=str, required=True, help="File containing states to solve")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory of nnet model")
    parser.add_argument('--env', type=str, required=True, help="Environment: cube3, 15-puzzle, 24-puzzle")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for BWAS")
    parser.add_argument('--weight', type=float, default=1.0, help="Weight of path cost")
    parser.add_argument('--inadmiss_tol', type=float, default=0.0, help="How much larger the cost of the computer path "
                                                                        "can be than cost of an optimal path")
    parser.add_argument('--results_file', type=str, required=True, help="File to save results")
    parser.add_argument('--start_idx', type=int, default=0, help="")
    parser.add_argument('--verbose', action='store_true', default=False, help="Set for verbose")

    args = parser.parse_args()

    # environment
    env: Environment = env_utils.get_environment(args.env)

    # get data
    input_data = pickle.load(open(args.states, "rb"))
    states: List[State] = input_data['states'][args.start_idx:]

    # initialize results
    results: Dict[str, Any] = dict()
    results["states"] = states
    results["solutions"] = []
    results["paths"] = []
    results["times"] = []
    results["num_nodes_generated"] = []

    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    heuristic_fn = nnet_utils.load_heuristic_fn(args.model_dir, device, on_gpu, env.get_nnet_model(),
                                                env, clip_zero=True)

    for state_idx, state in enumerate(states):
        start_time = time.time()

        num_itrs: int = 0
        astar = AStar([state], env, heuristic_fn, weight=args.weight)
        while not min(astar.has_found_goal()):
            astar.step(heuristic_fn, args.batch_size, verbose=args.verbose)
            num_itrs += 1

        path: List[State]
        soln: List[int]
        path_cost: float
        num_nodes_generated: int
        goal_node: Node = astar.get_goal_node_smallest_path_cost(0)
        path, soln, path_cost = get_path(goal_node)

        num_nodes_generated: int = astar.get_num_nodes_generated(0)

        solve_time = time.time() - start_time

        # record solution information
        results["solutions"].append(soln)
        results["paths"].append(path)
        results["times"].append(solve_time)
        # results["num_nodes_generated"].append(num_nodes_generated)

        # check soln
        assert search_utils.is_valid_soln(state, soln, env)

        # print to screen
        timing_str = ", ".join(["%s: %.2f" % (key, val) for key, val in astar.timings.items()])
        print("Times - %s, num_itrs: %i" % (timing_str, num_itrs))

        print("State: %i, SolnCost: %.2f, # Moves: %i, "
              "# Nodes Gen: %s, Time: %.2f" % (state_idx, path_cost, len(soln),
                                               format(num_nodes_generated, ","),
                                               solve_time))


if __name__ == "__main__":
    main()
