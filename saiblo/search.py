import math
from collections import defaultdict
from copy import deepcopy
from time import time


class TreeNode:
    __slots__ = ('children', 'N', 'Q', 'U', 'edge_P', 'state_hash')

    def __init__(self):
        self.children = {}
        self.N = 0
        self.Q = 0
        self.U = 0
        self.children = None # {action: (child_node, edge_N)}
        self.edge_P = None   # Dict[action, P]
        self.state_hash = None

    def select(self, c_puct, c_fpu):
        p_explored = sum(
            self.edge_P[action]
            for action, (_, edge_N) in self.children.items() if edge_N > 0
        )
        fpu_penalty = c_fpu * math.sqrt(p_explored)
        def uct(edge):
            action, (child, edge_N) = edge
            edge_Q = -child.Q if child else self.Q - fpu_penalty
            return edge_Q + c_puct * self.edge_P[action] * math.sqrt(self.N) / (1 + edge_N)
        return max(self.children.items(), key=uct)

    def update(self):
        self.Q = (self.U - sum(child.Q * edge_N for child, edge_N in self.children.values() if edge_N > 0)
                  ) / self.N


class MCGS:
    __slots__ = ('c_puct', 'c_fpu', 'policy', 'root', 'nodes_by_hash')

    def __init__(self, policy_value_fn, c_puct=1.0, c_fpu=0.2):
        self.c_puct = c_puct
        self.c_fpu = c_fpu
        self.policy = policy_value_fn
        self.root = None
        self.nodes_by_hash = defaultdict(TreeNode)

    def _playout(self, state):
        state = deepcopy(state)
        node = self.root
        path = []  # does not include leaf
        while node.children:
            path.append(node)
            action, (child, edge_N) = node.select(self.c_puct, self.c_fpu)
            state.step(action)
            if child is None:
                child = self.nodes_by_hash[state.hash]
            node.children[action] = (child, edge_N + 1)
            node = child
        if state.is_terminal():
            node.Q = -1.0 if state.winner else 0.0
        else:
            acts, probs, value = self.policy(state)
            node.edge_P = dict(zip(acts, probs))
            node.Q = node.U = value
            node.N = 1
            node.children = {action: (None, 0) for action in acts}
        for node in reversed(path):
            node.N += 1
            node.update()

    def _reroot(self, state):
        if state.hash not in self.nodes_by_hash:
            self.nodes_by_hash.clear()
            self.root = self.nodes_by_hash[state.hash]
            return
        self.root = self.nodes_by_hash[state.hash]
        for hash_value, node in self.nodes_by_hash.items():
            node.state_hash = hash_value
        self.nodes_by_hash.clear()
        def dfs(node):
            self.nodes_by_hash[node.state_hash] = node
            node.state_hash = None
            if not node.children:
                return
            for _, (child, _) in node.children.items():
                if child and child.state_hash:
                    dfs(child)
            node.update()
        dfs(self.root)

    def search(self, game, timeout):
        time_limit = time() + timeout
        self._reroot(game)
        # ensure at least 16 new playouts
        for _ in range(16):
            self._playout(game)
        while time() < time_limit:
            for _ in range(4):
                self._playout(game)
        return max(self.root.children.items(), key=lambda x: x[1][1])[0]
