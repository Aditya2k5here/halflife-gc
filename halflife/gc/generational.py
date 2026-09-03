"""Appel-style generational collector -- and the place where the experiment lives.

Structure: a bump-allocated nursery plus a mark-sweep old generation, a card-less
remembered set of old objects holding young pointers, and promotion on first
survival.

Everything in this project's headline result is one method: :meth:`choose_space`.
A stock generational collector allocates *everything* in the nursery and lets the
collector discover, expensively, that some of those objects were never going to
die. If you knew in advance which allocations produce survivors, you could put
them straight into the old generation and never copy them at all.

So this class takes a ``predictor`` -- an object that answers "will this
allocation still be alive when the nursery fills up?" -- and swaps it out:

    ``NeverPretenure``   stock generational collector (the baseline)
    ``PerfectOracle``    answers from recorded ground truth (the ceiling)
    ``SiteTable``        per-allocation-site empirical survival rate
    ``LearnedPredictor`` feature-hashed logistic regression

Because the four configurations differ *only* in that one predicate, any
difference in the numbers is attributable to prediction quality and nothing else.
"""

from .base import Collector
from ..vm.heap import SPACE_EDEN, SPACE_OLD


class Generational(Collector):
    name = "generational"
    generational = True

    def __init__(self, nursery_bytes=256 << 10, old_bytes=1 << 20,
                 predictor=None, old_growth=2.0, ground_truth=None, **cfg):
        super(Generational, self).__init__(
            heap_bytes=nursery_bytes + old_bytes, **cfg
        )
        self.nursery_bytes = nursery_bytes
        self.min_old_bytes = old_bytes
        self.old_threshold = old_bytes
        self.old_growth = old_growth
        self.predictor = predictor
        self.ground_truth = ground_truth

        self.eden = {}
        self.remset = set()
        self.peak_heap_bytes = nursery_bytes + old_bytes

        # prediction bookkeeping (measurement only -- never fed back into policy)
        self.n_predictions = 0
        self.pretenured_objects = 0
        self.pretenured_bytes = 0
        self.tp = self.fp = self.tn = self.fn = 0
        self.tp_bytes = self.fp_bytes = self.tn_bytes = self.fn_bytes = 0

        if predictor is not None and getattr(predictor, "name", None):
            self.name = "generational+" + predictor.name

    # --------------------------------------------------------------- policy
    def choose_space(self, site, size_words):
        pretenure = False
        if self.predictor is not None:
            pretenure = self.predictor.predict(site, size_words, self.vm)
        self.n_predictions += 1

        if self.ground_truth is not None:
            actual = self.ground_truth.survives(self.heap.alloc_seq)
            nbytes = size_words * 8
            if pretenure and actual:
                self.tp += 1
                self.tp_bytes += nbytes
            elif pretenure and not actual:
                self.fp += 1
                self.fp_bytes += nbytes
            elif not pretenure and actual:
                self.fn += 1
                self.fn_bytes += nbytes
            else:
                self.tn += 1
                self.tn_bytes += nbytes

        if pretenure:
            self.pretenured_objects += 1
            self.pretenured_bytes += size_words * 8
            return SPACE_OLD
        return SPACE_EDEN

    def should_collect(self, need_bytes, space):
        if space == SPACE_OLD:
            return self.heap.live_bytes[SPACE_OLD] + need_bytes > self.old_threshold
        return self.heap.live_bytes[SPACE_EDEN] + need_bytes > self.nursery_bytes

    def grow(self, need_bytes, space):
        if space == SPACE_OLD:
            self.old_threshold = max(
                self.old_threshold * 2,
                self.heap.live_bytes[SPACE_OLD] + need_bytes + 1,
            )
        else:
            self.nursery_bytes = max(
                self.nursery_bytes * 2,
                self.heap.live_bytes[SPACE_EDEN] + need_bytes + 1,
            )
        self.peak_heap_bytes = max(
            self.peak_heap_bytes, self.nursery_bytes + self.old_threshold
        )

    # ---------------------------------------------------------- bookkeeping
    def on_object_created(self, obj):
        if obj.space == SPACE_EDEN:
            self.eden[obj.addr] = obj
        else:
            obj.pretenured = True

    def write_barrier(self, obj, value):
        self.barrier_checks += 1
        if obj.space != SPACE_OLD:
            return
        if (value & 0b111) != 0:
            return
        child = self.heap.objects.get(value >> 3)
        if child is not None and child.space == SPACE_EDEN and obj not in self.remset:
            self.remset.add(obj)
            self.remset_adds += 1

    def post_init(self, obj):
        # A pretenured object is born old, so its initial field stores need the
        # same treatment a later store would have got from the write barrier.
        if obj.space != SPACE_OLD:
            return
        for v in obj.fields:
            self.write_barrier(obj, v)

    # ------------------------------------------------------------ collection
    def collect(self, reason="explicit", space=None):
        if space == SPACE_OLD or reason == "explicit":
            self.major(reason)
            return
        self.minor(reason)
        if self.heap.live_bytes[SPACE_OLD] > self.old_threshold:
            self.major("old-generation full")

    def minor(self, reason):
        self.n_collections += 1
        self.n_minor += 1
        before = self.heap.live_bytes[SPACE_EDEN]

        roots = self._roots()
        # Old-to-young pointers: the remembered set is scanned in full, which is
        # the price of the write barrier and is charged as tracing work.
        for obj in self.remset:
            self.bytes_traced += obj.size_bytes
            self.objects_traced += 1
            for v in obj.fields:
                if (v & 0b111) == 0:
                    roots.append(v)

        marked = self.mark_from(roots, into_space=SPACE_EDEN)

        promoted_bytes = 0
        for obj in self.eden.values():
            if obj.marked:
                self.heap.move_space(obj, SPACE_OLD)
                obj.age += 1
                promoted_bytes += obj.size_bytes
                self.objects_promoted += 1
            else:
                self.heap.free(obj)
        self.eden = {}
        self.clear_marks(marked)

        self.bytes_copied += promoted_bytes
        self.bytes_promoted += promoted_bytes
        self.remset.clear()
        self.record_event(
            "minor",
            {"reason": reason, "promoted": promoted_bytes, "freed": before - promoted_bytes},
        )

    def major(self, reason):
        self.n_collections += 1
        self.n_major += 1
        before = self.heap.total_live_bytes

        marked = self.mark_from(self._roots())
        self.bytes_swept += before

        promoted_bytes = 0
        for obj in list(self.heap.objects.values()):
            if not obj.marked:
                self.heap.free(obj)
            elif obj.space == SPACE_EDEN:
                self.heap.move_space(obj, SPACE_OLD)
                obj.age += 1
                promoted_bytes += obj.size_bytes
                self.objects_promoted += 1
        self.clear_marks(marked)

        self.bytes_copied += promoted_bytes
        self.bytes_promoted += promoted_bytes
        self.eden = {}
        self.remset.clear()

        old_live = self.heap.live_bytes[SPACE_OLD]
        self.old_threshold = max(self.min_old_bytes, int(old_live * self.old_growth))
        self.peak_heap_bytes = max(
            self.peak_heap_bytes, self.nursery_bytes + self.old_threshold
        )
        self.record_event(
            "major", {"reason": reason, "freed": before - self.heap.total_live_bytes}
        )

    # ------------------------------------------------------------------ stats
    def stats(self):
        s = super(Generational, self).stats()
        s.update({
            "nursery_bytes": self.nursery_bytes,
            "old_threshold": self.old_threshold,
            "predictions": self.n_predictions,
            "pretenured_objects": self.pretenured_objects,
            "pretenured_bytes": self.pretenured_bytes,
        })
        if self.ground_truth is not None:
            tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
            s.update({
                "pred_tp": tp, "pred_fp": fp, "pred_tn": tn, "pred_fn": fn,
                "pred_tp_bytes": self.tp_bytes, "pred_fp_bytes": self.fp_bytes,
                "pred_tn_bytes": self.tn_bytes, "pred_fn_bytes": self.fn_bytes,
                "pred_precision": tp / (tp + fp) if (tp + fp) else None,
                "pred_recall": tp / (tp + fn) if (tp + fn) else None,
                "pred_accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
                "survivor_rate": (tp + fn) / max(1, tp + tn + fp + fn),
            })
        return s
