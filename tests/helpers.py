"""Shared test utilities."""

from halflife.mite.compiler import compile_source
from halflife.vm.vm import VM
from halflife.values import is_ref, as_ref


def run(src, collector, **kw):
    module = compile_source(src, "<test>")
    vm = VM(module, collector, **kw)
    vm.run()
    return vm


def assert_heap_consistent(vm):
    """Every reference reachable from a live object or a root must resolve.

    This is the invariant a collector bug breaks first, and it is worth checking
    directly rather than hoping a benchmark notices: a dangling pointer often
    produces a *plausible* wrong answer rather than a crash.
    """
    objects = vm.heap.objects
    for addr, obj in objects.items():
        assert obj.addr == addr, "object %r filed under address %d" % (obj, addr)
        for v in obj.fields:
            if is_ref(v):
                assert as_ref(v) in objects, (
                    "dangling reference %d in %r" % (as_ref(v), obj)
                )
    for v in vm.iter_roots():
        if is_ref(v):
            assert as_ref(v) in objects, "dangling root reference %d" % as_ref(v)


def assert_accounting_consistent(vm):
    """The collector's per-space byte counters must match the objects present."""
    actual = [0, 0, 0]
    for obj in vm.heap.objects.values():
        actual[obj.space] += obj.size_bytes
    assert actual == vm.heap.live_bytes, (
        "space accounting drifted: counters=%r actual=%r"
        % (vm.heap.live_bytes, actual)
    )
