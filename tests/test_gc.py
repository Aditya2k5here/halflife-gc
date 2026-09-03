"""Collector correctness.

The properties that matter are not "the numbers look good" but:

1. a program produces identical output under every collector;
2. no collector ever leaves a dangling reference;
3. the per-space byte counters agree with the objects actually present;
4. the collectors actually collect -- a heap that never reclaims would pass (1)
   through (3) trivially.
"""

import pytest

from halflife.gc.marksweep import MarkSweep
from halflife.gc.semispace import SemiSpace
from halflife.gc.generational import Generational
from halflife.gc.predictors import NeverPretenure, AlwaysPretenure, SiteTable
from halflife.vm.heap import SPACE_EDEN, SPACE_OLD
from halflife.values import is_ref, as_ref

from helpers import run, assert_heap_consistent, assert_accounting_consistent

# A program that mixes long-lived structure, short-lived garbage, mutation of
# old objects (exercising the write barrier) and list growth.
STRESS = """
fun node(v, next) { return record { v: v, next: next, pad: [v, v, v] }; }

fun build(n) {
  let head = nil;
  let i = 0;
  while (i < n) { head = node(i, head); i = i + 1; }
  return head;
}

fun total(head) {
  let s = 0;
  let cur = head;
  while (cur != nil) { s = s + cur.v; cur = cur.next; }
  return s;
}

let kept = build(200);
let sum = 0;
let round = 0;
while (round < 60) {
  let junk = build(40);
  sum = sum + total(junk);
  if (round % 10 == 0) {
    kept.next = node(round, kept.next);   # mutates an old object: write barrier
  }
  round = round + 1;
}
sum = sum + total(kept);
print(sum);
"""


def collectors():
    return [
        MarkSweep(heap_bytes=48 << 10),
        SemiSpace(heap_bytes=96 << 10),
        Generational(nursery_bytes=16 << 10, old_bytes=48 << 10, predictor=NeverPretenure()),
        Generational(nursery_bytes=16 << 10, old_bytes=48 << 10, predictor=AlwaysPretenure()),
        Generational(nursery_bytes=16 << 10, old_bytes=48 << 10,
                     predictor=SiteTable({0: 1.0, 2: 1.0}, 0.5)),
    ]


@pytest.mark.parametrize("collector", collectors(), ids=lambda c: c.name)
def test_output_is_identical_under_every_collector(collector):
    vm = run(STRESS, collector)
    assert vm.output == ["66850"]


@pytest.mark.parametrize("collector", collectors(), ids=lambda c: c.name)
def test_no_dangling_references(collector):
    vm = run(STRESS, collector)
    assert_heap_consistent(vm)


@pytest.mark.parametrize("collector", collectors(), ids=lambda c: c.name)
def test_space_accounting(collector):
    vm = run(STRESS, collector)
    assert_accounting_consistent(vm)


@pytest.mark.parametrize("collector", collectors(), ids=lambda c: c.name)
def test_collectors_actually_collect(collector):
    vm = run(STRESS, collector)
    assert collector.n_collections > 0
    # The live set at exit must be far smaller than everything ever allocated.
    assert vm.heap.total_live_bytes < vm.heap.bytes_allocated // 4


def test_garbage_is_reclaimed_exactly():
    """After a collection, exactly the unreachable objects are gone."""
    src = """
    let keep = record { a: 1 };
    let i = 0;
    while (i < 500) { let junk = record { b: i, c: [i] }; i = i + 1; }
    print(keep.a);
    """
    gc = MarkSweep(heap_bytes=8 << 10)
    vm = run(src, gc)
    assert gc.n_collections > 0
    assert vm.heap.objects_allocated > 1000
    # Between collections the heap legitimately holds up to a budget of garbage,
    # so force one more and check what is genuinely reachable: just `keep`.
    gc.collect("explicit")
    shapes = [o.shape for o in vm.heap.objects.values()]
    assert ("a",) in shapes, "the retained record was collected"
    # Four objects, not one. The loop body's `junk` slot is still holding the
    # final record (and its list header and backing array) because root scanning
    # is precise about *where* roots are, not about whether a local is still
    # live. Real collectors have exactly this behaviour, and it is why "my object
    # wasn't freed" is so often a stale stack slot rather than a leak.
    assert len(vm.heap.objects) == 4


def test_pretenuring_places_objects_in_the_old_generation():
    src = "let i = 0; while (i < 50) { let r = record { a: i }; i = i + 1; }"
    gc = Generational(nursery_bytes=8 << 10, old_bytes=64 << 10,
                      predictor=AlwaysPretenure())
    vm = run(src, gc)
    assert gc.pretenured_objects == vm.heap.objects_allocated
    assert vm.heap.live_bytes[SPACE_EDEN] == 0
    assert all(o.space == SPACE_OLD for o in vm.heap.objects.values())


def test_never_pretenure_promotes_only_survivors():
    gc = Generational(nursery_bytes=16 << 10, old_bytes=48 << 10,
                      predictor=NeverPretenure())
    vm = run(STRESS, gc)
    assert gc.pretenured_objects == 0
    assert gc.n_minor > 0
    # Promotion must be a strict subset of allocation: nothing is copied twice.
    assert 0 < gc.bytes_promoted < vm.heap.bytes_allocated


def test_write_barrier_catches_old_to_young_pointers():
    """Without a remembered set this program loses an object and crashes."""
    src = """
    let holder = record { slot: nil };
    let i = 0;
    while (i < 400) {
      holder.slot = record { v: i, pad: [i, i, i, i] };
      let junk = [i, i, i, i, i, i];
      i = i + 1;
    }
    print(holder.slot.v);
    """
    gc = Generational(nursery_bytes=4 << 10, old_bytes=16 << 10,
                      predictor=NeverPretenure())
    vm = run(src, gc)
    assert vm.output == ["399"]
    assert gc.remset_adds > 0
    assert_heap_consistent(vm)


def test_pretenured_objects_register_in_the_remembered_set():
    """A pretenured object is born old, so even its *initial* fields need the
    barrier -- a bug that only shows up when the object is filled with young
    references at construction time."""
    src = """
    let holder = record { slot: nil };
    let i = 0;
    while (i < 200) {
      holder.slot = record { v: i, kid: record { n: i } };
      let junk = [i, i, i, i];
      i = i + 1;
    }
    print(holder.slot.kid.n);
    """
    gc = Generational(nursery_bytes=4 << 10, old_bytes=16 << 10,
                      predictor=AlwaysPretenure())
    vm = run(src, gc)
    assert vm.output == ["199"]
    assert_heap_consistent(vm)


def test_heap_grows_rather_than_failing_when_the_live_set_exceeds_the_budget():
    src = """
    let xs = [];
    let i = 0;
    while (i < 2000) { push(xs, record { a: i, b: [i, i] }); i = i + 1; }
    print(len(xs));
    """
    gc = MarkSweep(heap_bytes=4 << 10)
    vm = run(src, gc)
    assert vm.output == ["2000"]
    assert gc.peak_heap_bytes > (4 << 10)


def test_cost_model_components_are_consistent():
    gc = SemiSpace(heap_bytes=64 << 10)
    run(STRESS, gc)
    s = gc.stats()
    assert s["bytes_copied"] > 0 and s["bytes_swept"] == 0
    assert s["gc_work"] == pytest.approx(
        s["bytes_traced"] + 2.0 * s["bytes_copied"] + 0.1 * s["bytes_swept"]
    )


def test_all_reachable_objects_survive_a_collection():
    """Reachability computed independently of the collector's own marking."""
    gc = Generational(nursery_bytes=8 << 10, old_bytes=32 << 10,
                      predictor=NeverPretenure())
    vm = run(STRESS, gc)

    seen = set()
    stack = [as_ref(v) for v in vm.iter_roots() if is_ref(v)]
    while stack:
        addr = stack.pop()
        if addr in seen:
            continue
        seen.add(addr)
        obj = vm.heap.objects.get(addr)
        assert obj is not None, "root-reachable object %d was collected" % addr
        stack.extend(as_ref(v) for v in obj.fields if is_ref(v))
