import itertools
from collections.abc import Iterator
from typing import Literal

import networkx as nx
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import GraphInstance

SamplingMethod = Literal["g2sat", "uniform"]


class SATGraph:
    def __init__(
        self,
        num_vars: int,
        node_type: npt.NDArray[np.int64],  # N
        edge_index: npt.NDArray[np.int64],  # 2 x E
        node_degree: npt.NDArray[np.int64],  # N
        sampling_method: SamplingMethod = "g2sat",
        allow_overlaps: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.num_vars = num_vars
        self.node_type = node_type
        self.edge_index = edge_index
        self.node_degree = node_degree
        self.sampling_method = sampling_method
        self.allow_overlaps = allow_overlaps
        self.rng = np.random.default_rng(random_state)

        self.clause_vars = []
        for clause in self.to_clauses():
            self.clause_vars.append({abs(v) - 1 for v in clause})

    @property
    def num_nodes(self) -> int:
        return self.node_type.shape[0]

    @property
    def num_clauses(self) -> int:
        return self.node_type.shape[0] - self.num_vars * 2

    @property
    def clause_degree(self) -> npt.NDArray[np.int64]:
        return self.node_degree[self.num_vars * 2 :]

    def is_3sat(self) -> bool:
        return bool((self.clause_degree == 3).all())

    def get_valid_merges(self) -> list[tuple[int, int]]:
        valid_pairs = []
        for i, j in self._unfiltered_sat3_pairs():
            if i == j:
                continue
            if self.allow_overlaps or not self.have_overlapping_vars(i, j):
                if i > j:
                    i, j = j, i
                valid_pairs.append((i, j))
        return valid_pairs

    def _unfiltered_sat3_pairs(self) -> Iterator[tuple[int, int]]:
        ii_cross, jj_cross, ii_intra = self._valid_sat3_pairs()
        if self.sampling_method == "uniform":
            return itertools.chain(
                itertools.product(ii_cross, jj_cross),
                itertools.combinations(ii_intra, r=2),
            )

        min_degree = np.min(self.clause_degree)
        min_clauses = (
            np.argwhere(self.clause_degree == min_degree).ravel() + self.num_vars * 2
        )
        first = self.rng.choice(min_clauses)

        ii_cross, jj_cross, ii_intra = self._valid_sat3_pairs()
        seconds = []
        if first in ii_cross:
            seconds.append(jj_cross)
        if first in jj_cross:
            seconds.append(ii_cross)
        if first in ii_intra:
            seconds.append(ii_intra)

        return (
            (first, second) for seconds_inner in seconds for second in seconds_inner
        )

    def count_valid_merges(self) -> int:
        ii_cross, jj_cross, ii_intra = self._valid_sat3_pairs()
        len_cross = len(ii_cross) * len(jj_cross)
        len_intra = len(ii_intra) * (len(ii_intra) - 1) // 2
        return len_cross + len_intra

    def sample_valid_merges_with_count(
        self, k: int
    ) -> tuple[list[tuple[int, int]], int]:
        valid_pairs = self.get_valid_merges()
        if len(valid_pairs) <= k:
            return valid_pairs, len(valid_pairs)
        idxs = self.rng.choice(len(valid_pairs), size=k, replace=False)
        sample = [valid_pairs[i] for i in idxs]
        return sample, len(valid_pairs)

    def sample_valid_merges(self, k: int) -> list[tuple[int, int]]:
        return self.sample_valid_merges_with_count(k)[0]

    def _valid_sat3_pairs(
        self,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        target_clauses = self.clause_degree.sum() // 3
        mask1 = self.clause_degree == 1
        mask2 = self.clause_degree == 2
        mask3 = 1 - mask1 - mask2

        ii_cross = np.argwhere(mask1).ravel()
        jj_cross = np.argwhere(mask2).ravel()

        if mask2.sum() + mask3.sum() >= target_clauses:
            # We already have all the SAT2 and SAT3 clauses allowed,
            # so the only possible merges are between SAT1 and SAT2
            # clauses (no SAT1 x SAT1 merge)
            ii_intra = np.zeros(0, dtype=ii_cross.dtype)
        else:
            # Otherwise, we can merge any SAT1 or SAT2 clause
            # with another SAT1 clause
            ii_intra = ii_cross

        # Index of the first clause node
        clause_offset = 2 * self.num_vars
        return (
            ii_cross + clause_offset,
            jj_cross + clause_offset,
            ii_intra + clause_offset,
        )

    def merge(self, i: int, j: int) -> None:
        assert i != j and self.node_type[i] == 2 and self.node_type[j] == 2

        # Deterministic choice of which node is removed (j)
        # and which node is updated (i)
        if j < i:
            i, j = j, i

        deg_i = self.node_degree[i]
        deg_j = self.node_degree[j]
        if deg_i + deg_j > 3:
            print(
                f"Warning: action ({i}, {j}) creates a clause with more than 3 literals"
                f" (original nodes had degrees {deg_i} and {deg_j})"
            )

        vars_i = self.clause_vars[i - 2 * self.num_vars]
        vars_j = self.clause_vars[j - 2 * self.num_vars]
        vars_i.update(vars_j)
        self.clause_vars.pop(j - 2 * self.num_vars)

        # Remove duplicate literals
        literals_i = self.edge_index[(self.edge_index == i)[::-1]]
        literals_j = self.edge_index[(self.edge_index == j)[::-1]]
        duplicates = list(set(literals_i) & set(literals_j))
        duplicate_mask = (self.edge_index == j) | np.isin(self.edge_index, duplicates)
        duplicate_literal_idxs = np.argwhere(duplicate_mask.all(axis=0)).flatten()
        self.edge_index = np.delete(self.edge_index, duplicate_literal_idxs, axis=1)

        self.edge_index[self.edge_index == j] = i
        self.edge_index[self.edge_index > j] -= 1
        self.node_degree[i] += self.node_degree[j] - len(duplicates)

        self.node_type = np.delete(self.node_type, j, axis=0)
        self.node_degree = np.delete(self.node_degree, j, axis=0)

    def have_overlapping_vars(self, i: int, j: int) -> bool:
        i -= self.num_vars * 2
        j -= self.num_vars * 2
        return len(self.clause_vars[i] & self.clause_vars[j]) > 0

    def to_clauses(self) -> list[list[int]]:
        clauses = [[] for _ in range(self.num_clauses)]
        for edge in range(self.edge_index.shape[1]):
            i, j = self.edge_index[:, edge]
            if i > j:
                i, j = j, i
            if i < 2 * self.num_vars and j >= 2 * self.num_vars:
                clause = j - 2 * self.num_vars
                literal = i + 1
                if literal > self.num_vars:
                    literal = -(literal - self.num_vars)

                # by default, literal is of type np.int64, which can
                # cause issues later on
                clauses[clause].append(int(literal))
        return clauses

    def to_graph_instance(self, compressed: bool = True) -> GraphInstance:
        if compressed:
            nodes = self.node_type
        else:
            nodes = np.zeros((self.node_type.shape[0], 3), dtype=np.float32)
            nodes[self.node_type == 0, 0] = 1
            nodes[self.node_type == 1, 1] = 1
            nodes[self.node_type == 2, 2] = 1

        # Add reverse edges to make the graph undirected
        edge_links = np.concatenate([self.edge_index, self.edge_index[::-1]], axis=1)

        return GraphInstance(nodes=nodes, edges=None, edge_links=edge_links)

    def to_nx(self) -> nx.Graph:
        g = nx.Graph()
        for i, name in enumerate(["pos_literal", "neg_literal", "clause"]):
            g.add_nodes_from(np.argwhere(self.node_type == i).ravel(), kind=name)

        for i in range(self.edge_index.shape[1]):
            n1, n2 = self.edge_index[:, i]
            g.add_edge(n1, n2)

        mapping = {}
        for i in range(self.num_vars):
            mapping[i] = f"x{i+1}/{i}"
            mapping[i + self.num_vars] = f"-x{i+1}/{i+self.num_vars}"
        for i in range(self.num_clauses):
            mapping[i + 2 * self.num_vars] = f"c{i+1}/{i+2*self.num_vars}"
        nx.relabel_nodes(g, mapping, copy=False)

        return g

    def plot_nx(self, g: nx.Graph | None = None):
        if g is None:
            g = self.to_nx()

        pos_raw = {}
        coeff_lit = (self.num_clauses - 1) / (2 * self.num_vars - 1)
        if coeff_lit == 0:
            coeff_lit = 1
        for i in range(self.num_vars):
            pos_raw[f"x{i+1}"] = [0, (-2 * i) * coeff_lit]
            pos_raw[f"-x{i+1}"] = [0, (-2 * i - 1) * coeff_lit]
        for i in range(self.num_clauses):
            pos_raw[f"c{i+1}"] = [1, -i]

        pos = {}
        for node in g.nodes:
            pos[node] = pos_raw[node.split("/")[0]]

        node_color = []
        for n in g.nodes:
            if "-x" in n:
                node_color.append("tab:orange")
            elif "x" in n:
                node_color.append("tab:green")
            else:
                node_color.append("tab:blue")

        nx.draw_networkx(g, pos=pos, node_color=node_color, node_size=700, font_size=11)

    def __repr__(self) -> str:
        node_type = self.node_type.tolist()
        edge_index = self.edge_index.tolist()
        node_degree = self.node_degree.tolist()
        return f"SATGraph({node_type=}, {edge_index=}, {node_degree=})"

    @staticmethod
    def from_template(template: npt.NDArray[np.int64], **kwargs) -> "SATGraph":
        num_literals = len(template)
        num_vars = num_literals // 2
        num_clauses = template.sum()

        assert num_literals % 2 == 0

        edge_index_literals = np.repeat(np.arange(num_literals), repeats=template)
        edge_index_clauses = num_literals + np.arange(num_clauses)
        edge_index = np.vstack([edge_index_literals, edge_index_clauses])

        pos_nodes = np.arange(num_vars)
        neg_nodes = pos_nodes + num_vars
        literal_edges = np.vstack([pos_nodes, neg_nodes])
        edge_index = np.concatenate([literal_edges, edge_index], axis=1)

        node_degree = np.concatenate([template, np.ones(num_clauses, dtype=np.int64)])

        # node_type == 0 (positive literal), 1 (negative literal) or 2 (clause)
        node_type = np.repeat(np.arange(3), [num_vars, num_vars, num_clauses])

        return SATGraph(num_vars, node_type, edge_index, node_degree, **kwargs)

    @staticmethod
    def sample_template(
        num_vars: int,
        num_clauses: int,
        rng: np.random.Generator | None = None,
    ) -> npt.NDArray[np.int64]:
        assert num_vars * 2 <= num_clauses

        # To generate a template, we consider an array of clauses,
        # each with a single literal. We sort the literals:
        #   c = [l0, l0, l1, l1, l1, l2, ...]
        #                ^i=2        ^i=5
        # and obtain the (num_literals-1) indices where the literal changes
        # (2, 5, ... in the example above).
        # We then compute the number of occurrences of each literal
        # (2-0, 5-2, ... in the example above)
        if rng is None:
            rng = np.random.default_rng()
        num_literals = num_vars * 2
        partition_points = 1 + rng.choice(
            num_clauses - 1, size=num_literals - 1, replace=False
        )
        partition_points.sort()

        template = np.diff(partition_points, prepend=0, append=num_clauses)

        assert template.sum() == num_clauses
        assert template.shape[0] == num_literals

        return template
