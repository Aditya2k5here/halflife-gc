"""Lifetime tracing, feature extraction and the predictor.

The most important test in this file is
``test_online_and_offline_features_are_identical``. Everything else here checks
that a component works; that one checks that the *experiment* is valid. If the
tokens the model is trained on differ in any way from the tokens the collector
scores at run time, every number in the results table is meaningless, and the
failure mode is silent.
"""

import pytest

from halflife.gc.generational import Generational
from halflife.gc.marksweep import MarkSweep
from halflife.gc.predictors import (
    NeverPretenure, AlwaysPretenure, PerfectOracle, SiteTable, LearnedPredictor,
)
from halflife.mite.compiler import compile_source
from halflife.oracle.features import (
    alloc_context, alloc_tokens, tokens_from_context, transferable, _log_bucket,
)
from halflife.oracle.model import HashedLogistic, binary_metrics, roc_auc
from halflife.oracle.trace import trace_program, GroundTruth, INF
from halflife.vm.vm import VM

PROGRAM = """
fun mk(v) { return record { v: v, tag: [v] }; }

let keep = [];
let i = 0;
while (i < 400) {
  let temp = mk(i);
  if (i % 10 == 0) { push(keep, mk(i)); }
  i = i + 1;
}
print(len(keep));
"""


@pytest.fixture(scope="module")
def traced():
    module = compile_source(PROGRAM, "<test>")
    trace, vm = trace_program(module, granularity=512)
    return module, trace, vm


# --- tracing ---------------------------------------------------------------
def test_trace_records_one_row_per_allocation(traced):
    module, trace, vm = traced
    assert len(trace) == vm.heap.objects_allocated
    assert len(trace.contexts) == len(trace.birth) == len(trace.death) == len(trace.size)


def test_tracing_does_not_change_program_behaviour(traced):
    module, trace, vm = traced
    plain = VM(module, MarkSweep(heap_bytes=1 << 20))
    plain.run()
    assert plain.output == vm.output == ["40"]


def test_lifetimes_are_non_negative_and_immortals_are_infinite(traced):
    _, trace, _ = traced
    assert all(trace.lifetime(i) >= 0 for i in range(len(trace)))
    immortal = [i for i in range(len(trace)) if trace.death[i] == INF]
    assert immortal, "the retained list should still be alive at exit"


def test_survivors_are_a_minority_of_this_program(traced):
    _, trace, _ = traced
    s = trace.summary(4096)
    assert 0.0 < s["survivor_rate"] < 0.5


def test_higher_thresholds_select_fewer_objects(traced):
    _, trace, _ = traced
    counts = [sum(trace.labels(t)) for t in (0, 1024, 8192, 65536)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == len(trace)   # threshold 0 selects everything


def test_finer_granularity_never_reports_longer_lifetimes():
    """Coarse checkpoints over-estimate lifetimes; they must never under-estimate."""
    module = compile_source(PROGRAM, "<test>")
    coarse, _ = trace_program(module, granularity=4096)
    fine, _ = trace_program(compile_source(PROGRAM, "<test>"), granularity=256)
    assert len(coarse) == len(fine)
    slack = sum(
        1 for i in range(len(fine))
        if coarse.death[i] < fine.death[i] - 4096
    )
    assert slack == 0


# --- features --------------------------------------------------------------
def test_log_bucket():
    assert [_log_bucket(n) for n in (0, 1, 2, 3, 4, 7, 8)] == [0, 1, 2, 2, 3, 3, 4]


def test_transferable_drops_program_specific_tokens():
    toks = ("bias", "site=3", "fn=1", "caller=0", "kind=record", "depth=2")
    assert transferable(toks) == ("bias", "kind=record", "depth=2")


def test_online_and_offline_features_are_identical(traced):
    """Train/serve skew check.

    The tracer records a compact context per allocation and the trainer expands
    it into tokens later; the collector builds tokens directly from live VM
    state. Those two paths must agree exactly, allocation for allocation.
    """
    module, trace, _ = traced
    offline = [tokens_from_context(c, module) for c in trace.contexts]

    online = []

    class Recorder(NeverPretenure):
        def predict(self, site_id, size_words, vm):
            online.append(alloc_tokens(site_id, size_words, vm))
            return False

    replay = VM(compile_source(PROGRAM, "<test>"),
                Generational(nursery_bytes=8 << 10, old_bytes=32 << 10,
                             predictor=Recorder()))
    replay.run()

    assert len(online) == len(offline)
    assert online == offline


def test_allocation_sequence_is_identical_under_every_policy():
    """Trace-driven prediction only works if allocation #n is the same object in
    every run. Pretenuring changes GC timing, so this is worth asserting rather
    than assuming."""
    sequences = []
    for predictor in (NeverPretenure(), AlwaysPretenure()):
        seen = []

        class Recorder(type(predictor)):
            def predict(self, site_id, size_words, vm):
                seen.append(alloc_context(site_id, size_words, vm))
                return type(predictor).predict(self, site_id, size_words, vm)

        vm = VM(compile_source(PROGRAM, "<test>"),
                Generational(nursery_bytes=4 << 10, old_bytes=16 << 10,
                             predictor=Recorder()))
        vm.run()
        sequences.append(seen)
    assert sequences[0] == sequences[1]


# --- model -----------------------------------------------------------------
def test_model_learns_a_separable_signal():
    samples = [("bias", "site=1") if i % 2 else ("bias", "site=2") for i in range(600)]
    labels = [1 if s[1] == "site=1" else 0 for s in samples]
    model = HashedLogistic(dim=1024, epochs=12)
    history = model.fit(samples, labels)
    assert history[-1] < history[0]
    assert model.score(("bias", "site=1")) > 0.9
    assert model.score(("bias", "site=2")) < 0.1


def test_model_hashing_is_process_stable():
    a = HashedLogistic(dim=64)
    assert a._idx("site=7") == HashedLogistic(dim=64)._idx("site=7")


def test_model_roundtrips_through_json(tmp_path):
    model = HashedLogistic(dim=256, epochs=4)
    model.fit([("bias", "a"), ("bias", "b")], [1, 0])
    path = tmp_path / "m.json"
    model.save(str(path))
    loaded = HashedLogistic.load(str(path))
    # Weights are rounded to six decimals on the way out, so the round trip is
    # lossy by design -- a saved model is a few kilobytes instead of a megabyte.
    assert loaded.score(("bias", "a")) == pytest.approx(
        model.score(("bias", "a")), abs=1e-5
    )


def test_metrics():
    m = binary_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 2, 0)
    assert m["precision"] == 1.0 and m["recall"] == 0.5
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5
    assert roc_auc([1, 1], [0.1, 0.2]) is None


# --- predictors ------------------------------------------------------------
def test_perfect_oracle_matches_the_ground_truth_exactly(traced):
    module, trace, _ = traced
    gt = GroundTruth(trace, 8 << 10)
    gc = Generational(nursery_bytes=8 << 10, old_bytes=32 << 10,
                      predictor=PerfectOracle(gt), ground_truth=gt)
    vm = VM(compile_source(PROGRAM, "<test>"), gc)
    vm.run()
    stats = gc.stats()
    assert stats["pred_precision"] == 1.0
    assert stats["pred_recall"] == 1.0
    assert stats["pred_fp"] == 0 and stats["pred_fn"] == 0


def test_site_table_is_built_from_the_trace(traced):
    module, trace, _ = traced
    table = SiteTable.from_trace(trace, 8 << 10, threshold=0.5)
    assert table.rates
    assert all(0.0 <= r <= 1.0 for r in table.rates.values())
    assert table.describe()["sites"] == len(table.rates)


def test_learned_predictor_caches_by_feature_vector(traced):
    module, trace, _ = traced
    model = HashedLogistic(dim=1024)
    pred = LearnedPredictor(model, 0.5)
    gc = Generational(nursery_bytes=8 << 10, old_bytes=32 << 10, predictor=pred)
    VM(compile_source(PROGRAM, "<test>"), gc).run()
    # Thousands of allocations, a handful of distinct contexts.
    assert pred.lookups > 1000
    assert pred.misses < pred.lookups // 10
