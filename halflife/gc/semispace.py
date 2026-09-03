"""Cheney-style semispace copying collector.

Half the heap is unusable at any moment; collection evacuates the live set from
from-space to to-space and flips. Cost is proportional to the *live* data rather
than the whole heap, which is the classic argument for copying collection: if
most objects die, you never touch them.

In this simulator "copying" charges ``COPY_COST`` per live byte instead of
physically relocating anything -- see the note at the top of ``vm/heap.py`` for
why, and what that does and does not model.
"""

from .base import Collector
from ..vm.heap import SPACE_SINGLE


class SemiSpace(Collector):
    name = "semispace"

    def __init__(self, heap_bytes=4 << 20, **cfg):
        super(SemiSpace, self).__init__(heap_bytes=heap_bytes, **cfg)
        self.semi = heap_bytes // 2
        self.peak_heap_bytes = self.semi * 2

    def choose_space(self, site, size_words):
        return SPACE_SINGLE

    def should_collect(self, need_bytes, space):
        return self.heap.live_bytes[SPACE_SINGLE] + need_bytes > self.semi

    def grow(self, need_bytes, space):
        live = self.heap.live_bytes[SPACE_SINGLE]
        self.semi = max(self.semi * 2, live + need_bytes + 1)
        self.peak_heap_bytes = max(self.peak_heap_bytes, self.semi * 2)

    def collect(self, reason="explicit", space=None):
        self.n_collections += 1
        self.n_major += 1
        before = self.heap.live_bytes[SPACE_SINGLE]

        marked = self.mark_from(self._roots())
        survivors = sum(o.size_bytes for o in marked)
        for obj in list(self.heap.objects.values()):
            if not obj.marked:
                self.heap.free(obj)
        self.clear_marks(marked)
        # No sweep: dead objects are simply left behind in from-space. That is
        # the other half of the copying-collector bargain.

        # every survivor is evacuated to the other semispace
        self.bytes_copied += survivors
        self.record_event(
            "copy", {"reason": reason, "copied": survivors, "freed": before - survivors}
        )
