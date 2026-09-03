"""Capture a "heap film": the event tape needed to animate a run.

The tape is deliberately raw -- births, promotions, deaths and collection
markers, in order -- so that the HTML player can replay the heap rather than
show a chart of it. Watching a nursery fill with objects, flash, and empty,
while a handful of survivors migrate into the old generation, communicates what
a generational collector *is* in about four seconds. The same information as a
table takes a paragraph and lands worse.
"""

from ..eval.run import build, CONFIG, gen
from ..gc.predictors import NeverPretenure, PerfectOracle
from ..oracle.trace import trace_program, GroundTruth
from ..vm.vm import VM


def capture(name, predictor, ground_truth, max_events=45000, byte_limit=None):
    module = build(name, CONFIG["eval_variant"])
    vm = VM(module, gen(predictor, ground_truth),
            record_heap_events=True, max_events=max_events)
    vm.run()
    return {
        "workload": name,
        "policy": predictor.name,
        "nursery_bytes": CONFIG["nursery_bytes"],
        "events": vm.heap.events,
        "truncated": len(vm.heap.events) >= max_events,
        "total_bytes": vm.heap.bytes_allocated,
        "sites": [s.label() for s in module.sites],
    }


def capture_pair(name, k, max_events=45000):
    """Two films of the same program: stock generational, and perfect oracle.

    Same allocation sequence, same instant-by-instant object identities -- the
    only difference is where each object was born. Played side by side that is
    the entire thesis of the project in one picture.
    """
    module = build(name, CONFIG["eval_variant"])
    trace, _ = trace_program(module, CONFIG["trace_granularity"])
    gt = GroundTruth(trace, k * CONFIG["nursery_bytes"])
    return {
        "baseline": capture(name, NeverPretenure(), gt, max_events),
        "oracle": capture(name, PerfectOracle(gt), gt, max_events),
        "k": k,
    }
