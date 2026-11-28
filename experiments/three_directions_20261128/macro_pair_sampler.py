"""Train-global, equal-cell pair sampling for Direction A."""
import hashlib

import numpy as np

from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import CONDITIONS, rng

PAIR_COUNT = 2048


def ordered_ids_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


class MacroPairSampler:
    def __init__(self, dataset, domain, y, row_ids, train_identity):
        self.dataset = dataset
        self.train_identity = train_identity
        self.row_ids = np.array(row_ids, dtype="<i8", copy=True)
        self.row_ids.flags.writeable = False
        if (self.row_ids.shape != domain.shape or y.shape != (len(domain), 2)
                or len(np.unique(self.row_ids)) != len(domain)
                or not np.isin(domain, (0, 1, 2)).all() or not np.isin(y, (0, 1)).all()):
            raise ValueError("Expected canonical Train rows with unique IDs and binary task labels")

        joint = 2 * y[:, 0] + y[:, 1]
        # A pool's canonical offsets map its stable IDs back to the Train feature array.
        self._pools = {(d, a): np.flatnonzero((domain == d) & (joint == a))
                       for d in range(3) for a in range(4)}
        for pool in self._pools.values():
            pool.flags.writeable = False
        self.active = tuple((d, t, a, b) for d, t, a, b in CONDITIONS
                            if len(self._pools[d, a]) and len(self._pools[d, b]))
        cells = np.array([2 * d + t for d, t, _, _ in self.active], dtype=np.int64)
        if len(set(cells)) != 6:
            raise ValueError(f"{dataset} Train lacks one or more Macro6 task-cell supports")

        def relative_mass(d, t, a, b):
            sizes = [len(self._pools[d, state]) for state in range(4)]
            nplus = sum(n for state, n in enumerate(sizes) if (state >> (1 - t)) & 1)
            nminus = sum(sizes) - nplus
            return (sizes[a] / nplus) * (sizes[b] / nminus)

        r = np.array([relative_mass(*condition) for condition in self.active], dtype=np.float64)
        self.w = r / (6 * np.bincount(cells, weights=r, minlength=6)[cells])
        self.q = .5 / len(self.active) + .5 * self.w
        self._cdf = np.cumsum(self.q)
        for array in (self.w, self.q, self._cdf):
            array.flags.writeable = False
        self._identity = dict(dataset=dataset, train_identity=train_identity, pair_count=PAIR_COUNT,
                              train_row_ids_hash=ordered_ids_hash(self.row_ids),
                              pool_hashes={f"{d}:{a}": ordered_ids_hash(self.row_ids[pool])
                                           for (d, a), pool in self._pools.items()})

    def state_dict(self):
        return dict(self._identity, pool_hashes=dict(self._identity["pool_hashes"]))

    def sample_macro6(self, context):
        if context["dataset"] != self.dataset:
            raise ValueError("Macro6 context does not match the sampler dataset")
        u = rng(context, "macro6_condition").random()
        index = min(int(np.searchsorted(self._cdf, u, side="right")), len(self.active) - 1)
        d, t, a, b = self.active[index]
        gen = rng(context, "macro6_pair_indices")
        pos = gen.choice(self._pools[d, a], PAIR_COUNT, replace=True)
        neg = gen.choice(self._pools[d, b], PAIR_COUNT, replace=True)
        offsets = np.concatenate((pos, neg))
        ids = self.row_ids[offsets]
        return dict(condition=(d, t, a, b), w=float(self.w[index]), q=float(self.q[index]),
                    offsets=offsets, ids=ids, pair_index_hash=ordered_ids_hash(ids),
                    unique_pairs=len(np.unique(ids.reshape(2, PAIR_COUNT).T, axis=0)),
                    unique_endpoints=len(np.unique(ids)))
