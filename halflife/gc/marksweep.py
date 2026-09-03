"""Non-moving mark-sweep collector -- the baseline everything else is measured against.

One space, one budget. When the budget is exhausted we trace the whole live set
from the roots and delete everything we did not reach. There is no copying, so
``bytes_copied`` is always zero and all GC work is tracing work. That makes it
the cheapest collector per collection *for a small live set* and the most
expensive one for a large one, which is exactly the trade-off generational
collection exists to break.
"""

from .base import Collector
from ..vm.heap import SPACE_SINGLE


class MarkSweep(Collector):
    name = "mark-sweep"

    def __init__(self, heap_bytes=4 << 20, growth_load_factor=0.7, **cfg):
        super(MarkSweep, self).__init__(heap_bytes=heap_bytes, **cfg)
        self.budget = heap_bytes
        self.growth_load_factor = growth_load_factor
        self.peak_heap_bytes = self.budget

    def choose_space(self, site, size_words):
        return SPACE_SINGLE

    def should_collect(self, need_bytes, space):
        return self.heap.live_bytes[SPACE_SINGLE] + need_bytes > self.budget

    def grow(self, need_bytes, space):
        live = self.heap.live_bytes[SPACE_SINGLE]
        self.budget = max(self.budget * 2, int((live + need_bytes) / self.growth_load_factor) + 1)
        self.peak_heap_bytes = max(self.peak_heap_bytes, self.budget)

    def collect(self, reason="explicit", space=None):
        self.n_collections += 1
        self.n_major += 1
        before = self.heap.live_bytes[SPACE_SINGLE]

        marked = self.mark_from(self._roots())
        # The sweep walks the whole space in address order, live and dead alike.
        self.bytes_swept += before
        for obj in list(self.heap.objects.values()):
            if not obj.marked:
                self.heap.free(obj)
        self.clear_marks(marked)

        live = self.heap.live_bytes[SPACE_SINGLE]
        if live > self.budget * self.growth_load_factor:
            self.budget = int(live / self.growth_load_factor) + 1
            self.peak_heap_bytes = max(self.peak_heap_bytes, self.budget)
        self.record_event("major", {"reason": reason, "freed": before - live})
