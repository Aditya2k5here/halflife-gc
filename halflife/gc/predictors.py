"""Pretenuring policies.

Each predictor answers one question, on every allocation, before the object
exists: *will this object still be alive when the nursery next fills up?*

They are ordered here from "no information" to "all the information":

    ``NeverPretenure``   the assumption every production generational GC makes
    ``AlwaysPretenure``  the degenerate opposite, useful as a sanity bound
    ``SiteTable``        one number per allocation site, learned from a profile
    ``LearnedPredictor`` a model over site *and* calling context
    ``PerfectOracle``    reads the answer off the recorded trace

``PerfectOracle`` is not a proposal -- you cannot ship it. It exists to answer
the question that makes the rest of the numbers interpretable: *how much is
there to win at all?* Without it, a learned predictor that saves 4% looks
underwhelming; if the ceiling is 5%, the same 4% is most of what was available.
"""

from ..oracle.features import alloc_tokens


class Predictor(object):
    name = "never"

    def predict(self, site_id, size_words, vm):
        return False

    def describe(self):
        return {"name": self.name}


class NeverPretenure(Predictor):
    name = "never"


class AlwaysPretenure(Predictor):
    name = "always"

    def predict(self, site_id, size_words, vm):
        return True


class PerfectOracle(Predictor):
    """Cheats, on purpose, to establish the ceiling."""

    name = "oracle"

    def __init__(self, ground_truth):
        self.gt = ground_truth

    def predict(self, site_id, size_words, vm):
        return self.gt.survives(vm.heap.alloc_seq)

    def describe(self):
        return {"name": self.name, "objects": len(self.gt)}


class SiteTable(Predictor):
    """The obvious baseline: pretenure a site if most of its objects survive.

    This is roughly what the pretenuring literature of the early 2000s does, and
    it is the bar the learned model has to clear. If a linear model over context
    features cannot beat one number per site, then context does not matter for
    this workload -- which is itself worth reporting.
    """

    name = "sitetable"

    def __init__(self, rates, threshold=0.5):
        self.rates = rates
        self.threshold = threshold

    def predict(self, site_id, size_words, vm):
        return self.rates.get(site_id, 0.0) >= self.threshold

    def describe(self):
        return {
            "name": self.name,
            "threshold": self.threshold,
            "sites": len(self.rates),
            "sites_pretenured": sum(1 for r in self.rates.values() if r >= self.threshold),
        }

    @classmethod
    def from_trace(cls, trace, nursery_bytes, threshold=0.5):
        labels = trace.labels(nursery_bytes)
        tot = {}
        pos = {}
        for ctx, y in zip(trace.contexts, labels):
            s = ctx[0]
            tot[s] = tot.get(s, 0) + 1
            if y:
                pos[s] = pos.get(s, 0) + 1
        rates = {s: pos.get(s, 0) / n for s, n in tot.items()}
        return cls(rates, threshold)


class LearnedPredictor(Predictor):
    """Feature-hashed logistic regression, scored inside the allocator."""

    name = "learned"

    def __init__(self, model, threshold=0.5, token_filter=None, name=None):
        self.model = model
        self.threshold = threshold
        self.token_filter = token_filter
        if name:
            self.name = name
        # Memoise by feature vector. Allocation sites are hit millions of times
        # with a handful of distinct contexts, so this turns "score a model on
        # every allocation" into "one dict lookup on every allocation" -- which
        # is the difference between a policy you could ship and a thought
        # experiment.
        self._cache = {}
        self.lookups = 0
        self.misses = 0

    def predict(self, site_id, size_words, vm):
        toks = alloc_tokens(site_id, size_words, vm)
        if self.token_filter is not None:
            toks = self.token_filter(toks)
        self.lookups += 1
        hit = self._cache.get(toks)
        if hit is None:
            self.misses += 1
            hit = self.model.score(toks) >= self.threshold
            self._cache[toks] = hit
        return hit

    def describe(self):
        return {
            "name": self.name,
            "threshold": self.threshold,
            "distinct_contexts": len(self._cache),
            "score_calls": self.misses,
            "predictions": self.lookups,
            "nonzero_weights": sum(1 for w in self.model.w if w != 0.0),
            "dim": self.model.dim,
        }
