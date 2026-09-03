"""The experiment.

For each benchmark this runs the full pipeline:

    compile(profiling variant)  -> trace lifetimes -> train predictor
    compile(evaluation variant) -> trace lifetimes -> ground truth
    run every collector on the evaluation variant -> results table

Two methodological decisions are worth stating up front, because they are what
separates this from a benchmark that flatters itself:

**Train and test are different runs of the program.** The profiling variant and
the evaluation variant are the same source compiled with different injected
constants -- different input sizes, different random seeds, different data. A
model that had merely memorised "allocation #40912 survives" scores zero here.
This mirrors how profile-guided optimisation is actually deployed: profile once
on representative input, ship the profile, run on different input.

**Training is per-program, not global.** Allocation site ids are program-local,
so a model trained on one program cannot meaningfully be applied to another --
"site 7" is a different thing in each, and hashing them into the same feature
actively hurts. The `transfer` row exists to measure what is left when you strip
the program-specific features out and train one model across all benchmarks.

Everything measured here is deterministic: running it twice reproduces every counter
exactly. The only fields that move between runs are the wall-clock timings recorded
alongside them, and no result depends on those.
"""

import argparse
import json
import os
import time

from ..mite.compiler import compile_source
from ..vm.vm import VM
from ..gc.marksweep import MarkSweep
from ..gc.semispace import SemiSpace
from ..gc.generational import Generational
from ..gc.predictors import (
    NeverPretenure, AlwaysPretenure, PerfectOracle, SiteTable, LearnedPredictor,
)
from ..oracle.trace import trace_program, GroundTruth
from ..oracle.features import tokens_from_context, transferable
from ..oracle.model import HashedLogistic, binary_metrics, roc_auc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BENCH_DIR = os.path.join(ROOT, "bench")
RESULTS_DIR = os.path.join(ROOT, "results")

WORKLOADS = [
    "binarytrees",
    "lru_cache",
    "particles",
    "graph_bfs",
    "doc_pipeline",
]

CONFIG = {
    "nursery_bytes": 32 << 10,
    "old_bytes": 128 << 10,
    "flat_heap_bytes": 256 << 10,
    "trace_granularity": 1024,
    "model": {"dim": 1 << 14, "lr": 0.15, "l2": 1e-6, "epochs": 10, "pos_weight": 3.0},
    "learned_thresholds": [0.3, 0.5, 0.7, 0.9],
    "sitetable_thresholds": [0.3, 0.5, 0.7],
    "profile_variant": {"seed": 1, "scale": 0},
    "eval_variant": {"seed": 2, "scale": 1},
}


# --------------------------------------------------------------------- helpers
def build(name, variant):
    """Compile one variant of a benchmark.

    The prologue injects the two knobs and nothing else. Critically it contains
    no allocation sites, so site ids are identical between the two variants --
    which is what lets a profile gathered on one be applied to the other.
    """
    src = open(os.path.join(BENCH_DIR, name + ".mite")).read()
    prologue = "let seed = %d;\nlet scale = %d;\n" % (variant["seed"], variant["scale"])
    return compile_source(prologue + src, name + ".mite")


def run_collector(module, collector, expect_output=None):
    vm = VM(module, collector)
    t0 = time.time()
    vm.run()
    wall = time.time() - t0
    if expect_output is not None and vm.output != expect_output:
        raise AssertionError(
            "collector %s changed program output!\n  expected %r\n  got      %r"
            % (collector.name, expect_output, vm.output)
        )
    stats = collector.stats()
    stats["wall_seconds"] = round(wall, 3)
    stats["instructions"] = vm.instructions
    stats["objects_allocated"] = vm.heap.objects_allocated
    stats["bytes_allocated"] = vm.heap.bytes_allocated
    return stats, vm


def gen(predictor, ground_truth):
    return Generational(
        nursery_bytes=CONFIG["nursery_bytes"],
        old_bytes=CONFIG["old_bytes"],
        predictor=predictor,
        ground_truth=ground_truth,
    )


def train_model(samples, labels, cfg=None):
    cfg = dict(CONFIG["model"], **(cfg or {}))
    model = HashedLogistic(**cfg)
    model.fit(samples, labels)
    return model


def dataset(trace, target_bytes, token_filter=None):
    labels = trace.labels(target_bytes)
    samples = [tokens_from_context(c, trace.module) for c in trace.contexts]
    if token_filter is not None:
        samples = [token_filter(s) for s in samples]
    return samples, labels


def offline_metrics(model, samples, labels, threshold):
    scores = [model.score(s) for s in samples]
    preds = [1 if s >= threshold else 0 for s in scores]
    m = binary_metrics(labels, preds)
    m["auc"] = roc_auc(labels, scores)
    m["threshold"] = threshold
    return m


# ------------------------------------------------------------------ experiment
#
# The policy family
# -----------------
# Pretenuring is a family of policies parameterised by one number: pretenure an
# object iff it will live for at least ``K`` nurseries. ``K = 0`` pretenures
# everything, ``K = infinity`` pretenures nothing, and the stock generational
# collector is the latter. Sweeping ``K`` matters because "survives one nursery"
# turns out to be the *wrong* target: an object that outlives the nursery by a
# hair gets promoted and dies immediately afterwards, so pretenuring it just
# moves the garbage somewhere more expensive to collect.
#
# The oracle is allowed hindsight twice over -- it reads true lifetimes *and*
# picks its ``K`` on the run it is being scored on -- because its only job is to
# answer "how much is there to win?". The learned predictor gets neither: its
# ``K``, its decision threshold and its weights all come from the profiling run.

K_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def oracle_sweep(module, trace, nursery, expect_output=None):
    """Run the whole Oracle(K) family and return every point plus the best."""
    curve = []
    best = None
    for k in K_VALUES:
        gt = GroundTruth(trace, k * nursery)
        stats, _ = run_collector(
            module, gen(PerfectOracle(gt), gt), expect_output=expect_output
        )
        curve.append({"k": k, "gc_work": stats["gc_work"],
                      "pretenured_bytes": stats["pretenured_bytes"]})
        if best is None or stats["gc_work"] < best[1]["gc_work"]:
            best = (k, stats)
    return curve, best


def evaluate_workload(name, log=print):
    nursery = CONFIG["nursery_bytes"]
    log("\n=== %s ===" % name)

    profile_mod = build(name, CONFIG["profile_variant"])
    eval_mod = build(name, CONFIG["eval_variant"])

    t0 = time.time()
    profile_trace, _ = trace_program(profile_mod, CONFIG["trace_granularity"])
    eval_trace, eval_vm = trace_program(eval_mod, CONFIG["trace_granularity"])
    expected_output = eval_vm.output
    log("  traced in %.1fs  profile=%d objects  eval=%d objects"
        % (time.time() - t0, len(profile_trace), len(eval_trace)))

    # --- 1. the ceiling: best Oracle(K), chosen with full hindsight ----------
    eval_curve, (k_oracle, oracle_stats) = oracle_sweep(
        eval_mod, eval_trace, nursery, expected_output
    )
    log("  oracle family     best K=%.2f  work=%.0f" % (k_oracle, oracle_stats["gc_work"]))

    # --- 2. everything the learner is allowed to see: the profiling run ------
    profile_curve, (k_star, _) = oracle_sweep(profile_mod, profile_trace, nursery)
    log("  learner target    K*=%.2f (chosen on the profiling run)" % k_star)

    target_bytes = k_star * nursery
    tr_samples, tr_labels = dataset(profile_trace, target_bytes)
    te_samples, te_labels = dataset(eval_trace, target_bytes)

    pos = sum(tr_labels) or 1
    neg = len(tr_labels) - pos or 1
    pos_weight = min(10.0, max(1.0, neg / pos))
    model = train_model(tr_samples, tr_labels, {"pos_weight": pos_weight})

    gt_profile = GroundTruth(profile_trace, target_bytes)
    gt_eval = GroundTruth(eval_trace, target_bytes)

    def sweep(make_predictor, thresholds, label):
        best, rows = None, []
        for thr in thresholds:
            stats, _ = run_collector(profile_mod, gen(make_predictor(thr), gt_profile))
            rows.append({"threshold": thr, "gc_work": stats["gc_work"]})
            if best is None or stats["gc_work"] < best[1]:
                best = (thr, stats["gc_work"])
        log("  %-10s threshold sweep -> %.2f" % (label, best[0]))
        return best[0], rows

    learned_thr, learned_sweep = sweep(
        lambda t: LearnedPredictor(model, t), CONFIG["learned_thresholds"], "learned"
    )
    site_thr, site_sweep = sweep(
        lambda t: SiteTable.from_trace(profile_trace, target_bytes, t),
        CONFIG["sitetable_thresholds"], "sitetable",
    )

    # --- 3. score everything on the held-out variant -------------------------
    configs = [
        ("mark-sweep", lambda: MarkSweep(heap_bytes=CONFIG["flat_heap_bytes"])),
        ("semispace", lambda: SemiSpace(heap_bytes=CONFIG["flat_heap_bytes"])),
        ("generational", lambda: gen(NeverPretenure(), gt_eval)),
        ("gen+always", lambda: gen(AlwaysPretenure(), gt_eval)),
        ("gen+sitetable",
         lambda: gen(SiteTable.from_trace(profile_trace, target_bytes, site_thr), gt_eval)),
        ("gen+learned", lambda: gen(LearnedPredictor(model, learned_thr), gt_eval)),
    ]

    rows = []
    for label, factory in configs:
        collector = factory()
        stats, _ = run_collector(eval_mod, collector, expect_output=expected_output)
        stats["label"] = label
        if getattr(collector, "predictor", None) is not None:
            stats["predictor"] = collector.predictor.describe()
        rows.append(stats)

    oracle_stats["label"] = "gen+oracle"
    oracle_stats["predictor"] = {"name": "oracle", "k": k_oracle}
    rows.append(oracle_stats)

    for r in rows:
        log("    %-15s work=%12.0f  minor=%3d major=%3d copied=%9d pretenured=%s"
            % (r["label"], r["gc_work"], r["minor_collections"],
               r["major_collections"], r["bytes_copied"],
               r.get("pretenured_bytes", "-")))

    by_label = {r["label"]: r for r in rows}
    baseline = by_label["generational"]["gc_work"]
    ceiling = by_label["gen+oracle"]["gc_work"]
    headroom = baseline - ceiling

    def recovered(label):
        if headroom <= 0:
            return None
        return (baseline - by_label[label]["gc_work"]) / headroom

    log("    headroom %.0f  ->  learned recovers %s, site table %s"
        % (headroom,
           "n/a" if recovered("gen+learned") is None else "%.0f%%" % (100 * recovered("gen+learned")),
           "n/a" if recovered("gen+sitetable") is None else "%.0f%%" % (100 * recovered("gen+sitetable"))))

    return {
        "name": name,
        "profile_trace": profile_trace.summary(nursery),
        "eval_trace": eval_trace.summary(nursery),
        "n_sites": len(eval_mod.sites),
        "sites": [s.to_json() for s in eval_mod.sites],
        "policy_family": {
            "k_values": K_VALUES,
            "eval_curve": eval_curve,
            "profile_curve": profile_curve,
            "k_oracle": k_oracle,
            "k_star": k_star,
            "target_bytes": target_bytes,
        },
        "thresholds": {
            "learned": learned_thr, "learned_sweep": learned_sweep,
            "sitetable": site_thr, "sitetable_sweep": site_sweep,
        },
        "model": {
            "nonzero_weights": sum(1 for w in model.w if w != 0.0),
            "dim": model.dim,
            "pos_weight": pos_weight,
            "train_metrics": offline_metrics(model, tr_samples, tr_labels, learned_thr),
            "test_metrics": offline_metrics(model, te_samples, te_labels, learned_thr),
        },
        "collectors": rows,
        "baseline_work": baseline,
        "ceiling_work": ceiling,
        "headroom": headroom,
        "recovered": {
            "gen+learned": recovered("gen+learned"),
            "gen+sitetable": recovered("gen+sitetable"),
            "gen+always": recovered("gen+always"),
        },
        "_train": (tr_samples, tr_labels),
        "_eval": (eval_mod, profile_mod, gt_profile, gt_eval, expected_output,
                  baseline, ceiling),
    }


def run_transfer_experiment(per_workload, log=print):
    """One model, trained across every benchmark, using only the features that
    mean the same thing in every program.

    Allocation site ids, function ids and caller ids are stripped, leaving
    allocation kind, object size, call depth and how many objects this site has
    already produced. The question is whether allocation behaviour has any
    program-independent structure at all, or whether every program needs its own
    profile. Thresholds are still chosen on each program's profiling run, so
    this is a fair comparison against the per-program model.
    """
    samples, labels = [], []
    for res in per_workload:
        s, y = res["_train"]
        samples.extend(transferable(t) for t in s)
        labels.extend(y)
    pos = sum(labels) or 1
    pos_weight = min(10.0, max(1.0, (len(labels) - pos) / pos))
    log("\n=== transfer model: %d samples across %d programs ==="
        % (len(samples), len(per_workload)))
    model = train_model(samples, labels, {"pos_weight": pos_weight})

    out = {}
    for res in per_workload:
        (eval_mod, profile_mod, gt_profile, gt_eval,
         expected, baseline, ceiling) = res["_eval"]
        best_thr, best_work = None, None
        for thr in CONFIG["learned_thresholds"]:
            pred = LearnedPredictor(model, thr, token_filter=transferable, name="transfer")
            stats, _ = run_collector(profile_mod, gen(pred, gt_profile))
            if best_work is None or stats["gc_work"] < best_work:
                best_thr, best_work = thr, stats["gc_work"]
        pred = LearnedPredictor(model, best_thr, token_filter=transferable, name="transfer")
        stats, _ = run_collector(eval_mod, gen(pred, gt_eval), expect_output=expected)
        stats["label"] = "gen+transfer"
        stats["predictor"] = pred.describe()
        headroom = baseline - ceiling
        rec = (baseline - stats["gc_work"]) / headroom if headroom > 0 else None
        out[res["name"]] = {"stats": stats, "threshold": best_thr, "recovered": rec}
        log("  %-14s work=%12.0f  recovered=%s"
            % (res["name"], stats["gc_work"],
               "n/a" if rec is None else "%.0f%%" % (100 * rec)))
    return {
        "nonzero_weights": sum(1 for w in model.w if w != 0.0),
        "train_samples": len(samples),
        "per_workload": out,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="run the halflife-gc experiment")
    ap.add_argument("--workloads", nargs="*", default=WORKLOADS)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "results.json"))
    ap.add_argument("--skip-transfer", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    started = time.time()

    per_workload = [evaluate_workload(w) for w in args.workloads]
    transfer = None if args.skip_transfer else run_transfer_experiment(per_workload)

    for res in per_workload:
        del res["_train"]
        del res["_eval"]

    payload = {
        "config": CONFIG,
        "k_values": K_VALUES,
        "generated_seconds": round(time.time() - started, 1),
        "workloads": per_workload,
        "transfer": transfer,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("\nwrote %s in %.0fs" % (args.out, time.time() - started))
    return payload


if __name__ == "__main__":
    main()
