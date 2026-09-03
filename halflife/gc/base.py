"""Collector interface and shared tracing machinery.

Every collector in this project implements the same three-method contract:

    ``alloc(kind, size_words, site)``   -- may trigger a collection
    ``collect(reason)``                 -- reclaim
    ``stats()``                         -- deterministic event counters

and is measured with the same cost model:

    ``gc_work = bytes_traced + COPY_COST*bytes_copied + SWEEP_COST*bytes_swept``

Tracing a byte means reading an object header and its pointer slots. Copying a
byte means reading it *and* writing it elsewhere, plus fixing up the references
that pointed at it, so it is charged double. Sweeping a byte means walking past
an object header in address order without following pointers, which is cheap and
cache-friendly, so it is charged a tenth.

The exact constants are arguable. Two things make that survivable. First,
``results/results.json`` stores the raw component counters, so anyone who
disagrees can re-weight them without re-running anything. Second -- and more
importantly -- the headline experiment compares three collectors that share
*identical* machinery and differ only in where an object is first placed, so
their relative ordering is independent of these weights entirely. The constants
only matter for the sanity-check comparison against mark-sweep and semispace.
"""

from ..values import is_ref, as_ref
from ..vm.heap import WORD_BYTES, SPACE_SINGLE

COPY_COST = 2.0
SWEEP_COST = 0.1


class OutOfMemory(Exception):
    pass


class Collector(object):
    name = "base"
    generational = False

    def __init__(self, heap_bytes=4 << 20, **cfg):
        self.heap_bytes = heap_bytes
        self.cfg = cfg
        self.vm = None
        self.heap = None
        self.reset_stats()

    # ------------------------------------------------------------------ setup
    def attach(self, vm):
        self.vm = vm
        self.heap = vm.heap

    def reset_stats(self):
        self.n_collections = 0
        self.n_minor = 0
        self.n_major = 0
        self.bytes_traced = 0
        self.bytes_copied = 0
        self.bytes_swept = 0
        self.objects_traced = 0
        self.objects_promoted = 0
        self.bytes_promoted = 0
        self.barrier_checks = 0
        self.remset_adds = 0
        self.peak_live_bytes = 0
        self.peak_heap_bytes = 0
        self.gc_events = []

    # -------------------------------------------------------------- allocation
    def choose_space(self, site, size_words):
        return SPACE_SINGLE

    def should_collect(self, need_bytes, space):
        raise NotImplementedError

    def alloc(self, kind, size_words, site):
        need = size_words * WORD_BYTES
        # choose_space is called exactly once per allocation: it is where the
        # pretenuring decision (and its bookkeeping) happens.
        space = self.choose_space(site, size_words)
        if self.should_collect(need, space):
            self.collect("allocation", space)
            if self.should_collect(need, space):
                self.grow(need, space)
        obj = self.heap.new(kind, size_words, site, space)
        self.on_object_created(obj)
        live = self.heap.total_live_bytes
        if live > self.peak_live_bytes:
            self.peak_live_bytes = live
        return obj

    def grow(self, need_bytes, space):
        raise OutOfMemory(
            "%s: cannot satisfy %d bytes (live=%d, budget=%d)"
            % (self.name, need_bytes, self.heap.total_live_bytes, self.heap_bytes)
        )

    def on_object_created(self, obj):
        """Called the instant an object exists, before its fields are filled.

        Registration has to happen here rather than in ``post_init``: a single
        source-level operation can allocate twice (a list is a header plus a
        backing array), and the second allocation can trigger a collection that
        must already know about the first.
        """

    # ------------------------------------------------------------------ barrier
    def write_barrier(self, obj, value):
        """Called on every pointer store into an existing object."""
        self.barrier_checks += 1

    def post_init(self, obj):
        """Called once after a freshly allocated object's fields are filled."""

    # ------------------------------------------------------------------ tracing
    def _roots(self):
        return [v for v in self.vm.iter_roots() if is_ref(v)]

    def mark_from(self, root_values, into_space=None, clear=True):
        """Transitive mark. Returns the list of marked objects, in visit order.

        ``into_space`` restricts traversal: if set, the walk only *enters*
        objects in that space. That restriction is what makes a minor collection
        cheap -- it is the entire reason generational collection exists -- and
        the caller is responsible for supplying the remembered set as extra
        roots so that old-to-young pointers are not missed.
        """
        objects = self.heap.objects
        marked = []
        stack = []
        for v in root_values:
            a = as_ref(v)
            o = objects.get(a)
            if o is None or o.marked:
                continue
            if into_space is not None and o.space != into_space:
                continue
            o.marked = True
            marked.append(o)
            stack.append(o)
        while stack:
            o = stack.pop()
            self.bytes_traced += o.size_bytes
            self.objects_traced += 1
            for v in o.fields:
                if (v & 0b111) != 0:      # not a reference (int / nil / bool)
                    continue
                child = objects.get(v >> 3)
                if child is None or child.marked:
                    continue
                if into_space is not None and child.space != into_space:
                    continue
                child.marked = True
                marked.append(child)
                stack.append(child)
        if clear:
            pass  # caller clears; marks are needed for the sweep
        return marked

    def clear_marks(self, objs):
        for o in objs:
            o.marked = False

    # -------------------------------------------------------------------- stats
    def gc_work(self):
        return (
            self.bytes_traced
            + COPY_COST * self.bytes_copied
            + SWEEP_COST * self.bytes_swept
        )

    def stats(self):
        return {
            "collector": self.name,
            "collections": self.n_collections,
            "minor_collections": self.n_minor,
            "major_collections": self.n_major,
            "bytes_traced": self.bytes_traced,
            "bytes_copied": self.bytes_copied,
            "bytes_swept": self.bytes_swept,
            "objects_traced": self.objects_traced,
            "objects_promoted": self.objects_promoted,
            "bytes_promoted": self.bytes_promoted,
            "barrier_checks": self.barrier_checks,
            "remset_adds": self.remset_adds,
            "peak_live_bytes": self.peak_live_bytes,
            "peak_heap_bytes": self.peak_heap_bytes,
            "gc_work": self.gc_work(),
        }

    def record_event(self, kind, extra=None):
        ev = {
            "kind": kind,
            "at_bytes": self.heap.bytes_allocated,
            "live": self.heap.total_live_bytes,
        }
        if extra:
            ev.update(extra)
        self.gc_events.append(ev)
        # Interleave a marker into the heap's event tape so the visualiser can
        # replay births, promotions, deaths and collections in the right order.
        self.heap._record("g", kind, self.heap.bytes_allocated)
