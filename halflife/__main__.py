"""Command line entry point: ``python -m halflife <command>``."""

import argparse
import os
import sys

from .mite.compiler import compile_source
from .vm.vm import VM
from .gc.marksweep import MarkSweep
from .gc.semispace import SemiSpace
from .gc.generational import Generational
from .gc.predictors import NeverPretenure, AlwaysPretenure, PerfectOracle
from .oracle.trace import trace_program, GroundTruth

COLLECTORS = {
    "mark-sweep": lambda a: MarkSweep(heap_bytes=a.heap),
    "semispace": lambda a: SemiSpace(heap_bytes=a.heap),
    "generational": lambda a: Generational(nursery_bytes=a.nursery,
                                           old_bytes=a.heap,
                                           predictor=NeverPretenure()),
    "gen+always": lambda a: Generational(nursery_bytes=a.nursery,
                                         old_bytes=a.heap,
                                         predictor=AlwaysPretenure()),
}


def load(path, seed=1, scale=0):
    src = open(path).read()
    prologue = ""
    if "seed" in src or "scale" in src:
        prologue = "let seed = %d;\nlet scale = %d;\n" % (seed, scale)
    return compile_source(prologue + src, os.path.basename(path))


def cmd_run(args):
    module = load(args.file, args.seed, args.scale)
    if args.collector == "gen+oracle":
        trace, _ = trace_program(module, args.granularity)
        gt = GroundTruth(trace, args.nursery)
        collector = Generational(nursery_bytes=args.nursery, old_bytes=args.heap,
                                 predictor=PerfectOracle(gt), ground_truth=gt)
        module = load(args.file, args.seed, args.scale)
    else:
        collector = COLLECTORS[args.collector](args)
    vm = VM(module, collector)
    vm.run()
    for line in vm.output:
        print(line)
    if not args.quiet:
        s = collector.stats()
        print("---", file=sys.stderr)
        print("collector      %s" % s["collector"], file=sys.stderr)
        print("instructions   %d" % vm.instructions, file=sys.stderr)
        print("allocated      %d objects / %d bytes"
              % (vm.heap.objects_allocated, vm.heap.bytes_allocated), file=sys.stderr)
        print("collections    %d (minor %d, major %d)"
              % (s["collections"], s["minor_collections"], s["major_collections"]),
              file=sys.stderr)
        print("bytes traced   %d" % s["bytes_traced"], file=sys.stderr)
        print("bytes copied   %d" % s["bytes_copied"], file=sys.stderr)
        print("bytes swept    %d" % s["bytes_swept"], file=sys.stderr)
        print("GC work        %.0f" % s["gc_work"], file=sys.stderr)
    return 0


def cmd_dis(args):
    print(load(args.file, args.seed, args.scale).disassemble(args.function))
    return 0


def cmd_sites(args):
    module = load(args.file, args.seed, args.scale)
    trace, _ = trace_program(module, args.granularity)
    labels = trace.labels(args.nursery)
    per_site = {}
    for ctx, size, y in zip(trace.contexts, trace.size, labels):
        d = per_site.setdefault(ctx[0], [0, 0, 0])
        d[0] += 1
        d[1] += y
        d[2] += size
    print("%-46s %8s %8s %9s %s" % ("site", "objects", "survive", "bytes", "rate"))
    for site_id, (n, surv, nbytes) in sorted(
            per_site.items(), key=lambda kv: -kv[1][2]):
        print("%-46s %8d %8d %9d %5.1f%%"
              % (module.sites[site_id].label()[:46], n, surv, nbytes, 100.0 * surv / n))
    s = trace.summary(args.nursery)
    print("\n%d objects, %d bytes, %.1f%% survive a %dB nursery, %d immortal"
          % (s["objects"], s["bytes_allocated"], 100 * s["survivor_rate"],
             args.nursery, s["immortal"]))
    return 0


def cmd_bench(args):
    from .eval.run import main as run_main
    run_main([] if not args.workloads else ["--workloads"] + args.workloads)
    return 0


def cmd_report(args):
    from .viz.report import main as report_main
    report_main()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="halflife", description=__doc__)
    ap.add_argument("--heap", type=lambda s: int(s, 0), default=256 << 10,
                    help="heap / old-generation budget in bytes")
    ap.add_argument("--nursery", type=lambda s: int(s, 0), default=32 << 10)
    ap.add_argument("--granularity", type=lambda s: int(s, 0), default=1024,
                    help="lifetime tracing checkpoint interval in bytes")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="execute a Mite program")
    p.add_argument("file")
    p.add_argument("--collector", default="generational",
                   choices=sorted(list(COLLECTORS) + ["gen+oracle"]))
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("dis", help="disassemble a Mite program")
    p.add_argument("file")
    p.add_argument("--function", default=None)
    p.set_defaults(func=cmd_dis)

    p = sub.add_parser("sites", help="per-allocation-site lifetime profile")
    p.add_argument("file")
    p.set_defaults(func=cmd_sites)

    p = sub.add_parser("bench", help="run the full experiment")
    p.add_argument("--workloads", nargs="*")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("report", help="rebuild results/report.html and summary.md")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
