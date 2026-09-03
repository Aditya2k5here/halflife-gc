"""A feature-hashed logistic regression, implemented from scratch.

Why hand-rolled instead of scikit-learn:

* This model runs *inside the allocator*. Scoring happens on every single
  allocation, so it has to be a dictionary lookup and a few multiply-adds, not a
  numpy call with array-construction overhead per sample.
* The whole repository has zero runtime dependencies, so the experiment
  reproduces from a bare Python install a decade from now.
* Feature hashing with a fixed CRC32 keeps model files portable. Python's
  built-in ``hash()`` on strings is randomised per process, so a model trained in
  one process would silently score garbage in the next -- a genuinely nasty bug
  to find, and one worth designing out.

The model is deliberately simple. If a linear model on hashed allocation-site
features can already recover most of the achievable win, that is the interesting
result; reaching for a gradient-boosted forest first would have hidden it.
"""

import json
import math
from zlib import crc32


class HashedLogistic(object):
    def __init__(self, dim=1 << 14, lr=0.1, l2=1e-6, epochs=12,
                 pos_weight=1.0, seed=0):
        self.dim = dim
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.pos_weight = pos_weight
        self.seed = seed
        self.w = [0.0] * dim
        self.trained_on = 0

    # ----------------------------------------------------------------- hashing
    def _idx(self, token):
        return crc32(token.encode("utf-8")) % self.dim

    def _sign(self, token):
        # signed hashing: cancels a little of the collision bias for free
        return 1.0 if (crc32(("s:" + token).encode("utf-8")) & 1) else -1.0

    # ---------------------------------------------------------------- scoring
    def raw_score(self, tokens):
        w = self.w
        total = 0.0
        for t in tokens:
            total += w[self._idx(t)] * self._sign(t)
        return total

    def score(self, tokens):
        z = self.raw_score(tokens)
        if z < -35.0:
            return 0.0
        if z > 35.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))

    # --------------------------------------------------------------- training
    def fit(self, samples, labels, verbose=False):
        """``samples`` is a sequence of token tuples; ``labels`` a 0/1 sequence.

        Plain SGD with a deterministic shuffle. Deterministic because every
        number in this repository has to be reproducible from ``make results``.
        """
        n = len(samples)
        order = list(range(n))
        rng = _LCG(self.seed)
        history = []
        for epoch in range(self.epochs):
            rng.shuffle(order)
            loss = 0.0
            lr = self.lr / (1.0 + epoch * 0.3)
            for i in order:
                toks = samples[i]
                y = labels[i]
                p = self.score(toks)
                weight = self.pos_weight if y else 1.0
                g = (p - y) * weight
                eps = 1e-12
                loss += -weight * (
                    math.log(p + eps) if y else math.log(1.0 - p + eps)
                )
                for t in toks:
                    j = self._idx(t)
                    s = self._sign(t)
                    self.w[j] -= lr * (g * s + self.l2 * self.w[j])
            history.append(loss / max(1, n))
            if verbose:
                print("  epoch %2d  mean loss %.5f" % (epoch + 1, history[-1]))
        self.trained_on = n
        return history

    # ----------------------------------------------------------------- persist
    def to_json(self):
        nz = {str(i): round(v, 6) for i, v in enumerate(self.w) if v != 0.0}
        return {
            "dim": self.dim, "lr": self.lr, "l2": self.l2,
            "epochs": self.epochs, "pos_weight": self.pos_weight,
            "seed": self.seed, "trained_on": self.trained_on,
            "nonzero": len(nz), "weights": nz,
        }

    @classmethod
    def from_json(cls, d):
        m = cls(dim=d["dim"], lr=d["lr"], l2=d["l2"], epochs=d["epochs"],
                pos_weight=d.get("pos_weight", 1.0), seed=d.get("seed", 0))
        for k, v in d["weights"].items():
            m.w[int(k)] = v
        m.trained_on = d.get("trained_on", 0)
        return m

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_json(), fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_json(json.load(fh))


class _LCG(object):
    """Tiny deterministic PRNG so shuffling does not depend on Python's."""

    def __init__(self, seed):
        self.state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.state >> 33

    def shuffle(self, lst):
        for i in range(len(lst) - 1, 0, -1):
            j = self.next() % (i + 1)
            lst[i], lst[j] = lst[j], lst[i]


# ----------------------------------------------------------------- evaluation
def binary_metrics(y_true, y_pred):
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if p and t:
            tp += 1
        elif p and not t:
            fp += 1
        elif not p and t:
            fn += 1
        else:
            tn += 1
    n = max(1, tp + fp + tn + fn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / n,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "base_rate": (tp + fn) / n,
    }


def roc_auc(y_true, scores):
    """Rank-based AUC (Mann-Whitney U), with ties handled by average rank."""
    pairs = sorted(zip(scores, y_true))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(1 for _, y in pairs if y)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
