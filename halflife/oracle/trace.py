"""Ground-truth object lifetimes.

To ask "how much could a perfect predictor win?" you first have to know, for
every object the program allocated, exactly how long it stayed alive. That is
what this module produces.

**How.** The program is run under a collector that collects extremely often --
every ``granularity`` bytes of allocation -- and records, for each object, the
first checkpoint at which it was unreachable. Lifetime is measured in *bytes
allocated since birth*, not in seconds and not in collections, because that unit
is a property of the program alone: it does not move when you change the heap
size, the collector, or the machine.

**The error this introduces.** An object that dies just after a checkpoint is not
noticed until the next one, so lifetimes are over-estimated by up to
``granularity`` bytes. The default granularity is a small fraction of the nursery,
which bounds the error well below the decision boundary we care about. The exact
alternative is Hertz's Merlin algorithm, which reconstructs precise death times
from timestamps rather than re-tracing; it is listed in the README's future work
because it would let this run with granularity 1 at a fraction of the cost.

**Why this is legitimate rather than cheating.** The tracer's output is used two
ways. As an *upper bound* (the `PerfectOracle` collector) it is deliberately
unattainable information -- that is the point, it measures the size of the prize.
As *training labels* it is offline supervision from a profiling run, which is
exactly how profile-guided optimisation works in real compilers.
"""

from ..gc.base import Collector
from ..vm.heap import SPACE_SINGLE
from .features import alloc_context

INF = float("inf")


class Trace(object):
    """Recorded lifetimes for one program run."""

    def __init__(self, module, granularity):
        self.module = module
        self.granularity = granularity
        self.contexts = []      # per allocation seq
        self.birth = []
        self.death = []
        self.size = []
        self.total_bytes = 0
        self.n_checkpoints = 0

    def __len__(self):
        return len(self.birth)

    def lifetime(self, seq):
        return self.death[seq] - self.birth[seq]

    def labels(self, threshold_bytes):
        """1 if the object lives for at least ``threshold_bytes`` of allocation.

        The threshold is a parameter rather than "one nursery" because the right
        cut-off for *pretenuring* is not the same as the cut-off for *surviving*.
        An object that outlives the nursery by a hair gets promoted and then dies
        immediately: promoting it early would only have moved the garbage. The
        experiment sweeps this threshold rather than assuming it.
        """
        return [
            1 if (d - b) >= threshold_bytes else 0
            for b, d in zip(self.birth, self.death)
        ]

    def summary(self, nursery_bytes):
        lab = self.labels(nursery_bytes)  # survivor stats are quoted per nursery
        n = len(lab)
        surv = sum(lab)
        surv_bytes = sum(s for s, l in zip(self.size, lab) if l)
        immortal = sum(1 for d in self.death if d == INF)
        return {
            "objects": n,
            "bytes_allocated": self.total_bytes,
            "survivors": surv,
            "survivor_rate": surv / n if n else 0.0,
            "survivor_bytes": surv_bytes,
            "survivor_byte_rate": surv_bytes / max(1, sum(self.size)),
            "immortal": immortal,
            "checkpoints": self.n_checkpoints,
            "granularity": self.granularity,
        }


class LifetimeTracer(Collector):
    """A mark-sweep collector with an absurdly small budget and a notebook.

    Reusing the collector interface for this is not a trick: "find everything
    that is unreachable right now" is precisely a collection, so the tracer is a
    collector that happens to write down what it killed.
    """

    name = "lifetime-tracer"

    def __init__(self, granularity=4096, **cfg):
        super(LifetimeTracer, self).__init__(heap_bytes=1 << 62, **cfg)
        self.granularity = granularity
        self.last_checkpoint = 0
        self.trace = None

    def attach(self, vm):
        super(LifetimeTracer, self).attach(vm)
        self.trace = Trace(vm.module, self.granularity)

    def choose_space(self, site, size_words):
        return SPACE_SINGLE

    def should_collect(self, need_bytes, space):
        return self.heap.bytes_allocated - self.last_checkpoint >= self.granularity

    def grow(self, need_bytes, space):
        pass  # the tracer has no budget to exceed

    def observe(self, obj, vm):
        """Allocation observer: installed on the VM, records one row per object."""
        t = self.trace
        t.contexts.append(alloc_context(obj.site, obj.size_words, vm))
        t.birth.append(obj.birth_bytes)
        t.death.append(INF)
        t.size.append(obj.size_bytes)

    def collect(self, reason="explicit", space=None):
        self.last_checkpoint = self.heap.bytes_allocated
        self.n_collections += 1
        self.trace.n_checkpoints += 1

        marked = self.mark_from(self._roots())
        now = self.heap.bytes_allocated
        death = self.trace.death
        for obj in list(self.heap.objects.values()):
            if not obj.marked:
                # First checkpoint at which this object was unreachable.
                death[obj.seq] = now
                self.heap.free(obj)
        self.clear_marks(marked)

    def finish(self):
        self.trace.total_bytes = self.heap.bytes_allocated
        return self.trace


def trace_program(module, granularity=4096, max_instructions=200_000_000):
    """Run ``module`` under the tracer and return a :class:`Trace`."""
    from ..vm.vm import VM

    tracer = LifetimeTracer(granularity=granularity)
    vm = VM(module, tracer, max_instructions=max_instructions,
            alloc_observer=tracer.observe)
    vm.run()
    return tracer.finish(), vm


class GroundTruth(object):
    """Answers "does allocation #n live at least ``threshold_bytes``?" in O(1)."""

    def __init__(self, trace, threshold_bytes):
        self.threshold_bytes = threshold_bytes
        self._surv = trace.labels(threshold_bytes)

    def survives(self, seq):
        return bool(self._surv[seq]) if seq < len(self._surv) else False

    def __len__(self):
        return len(self._surv)
