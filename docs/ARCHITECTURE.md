# Architecture notes

Design decisions that are load-bearing, and the reasoning behind them. The README says what
the project does; this says why it is built the way it is.

---

## 1. Why a new language instead of instrumenting a real one

The predictor needs a stable identity for every allocation. In CPython you can observe that
*an* object was created, but attributing it to a source construct means fighting the
interpreter, and any instrumentation heavy enough to record lifetimes changes the timing you
are measuring. The alternative — generating synthetic allocation traces from a statistical
model — is worse: the allocation patterns become an artefact of the generator, and any
predictor trained on them is learning the generator.

Mite is the middle path. It is small enough to build in a day and real enough to have
recursion, aliasing, mutation of long-lived structures, and dynamic arrays that reallocate.
The compiler stamps every allocating expression with a site id, and that id survives
recompilation of a different input variant, which is what makes profile-guided evaluation
possible at all.

## 2. Values are integers, not objects

A Mite value is a Python `int` with three tag bits: `0b001` small integer, `0b010`
nil/true/false, `0b000` heap reference. This is not decoration. If values were Python objects
then running a Mite program would allocate in *CPython's* heap in proportion to Mite's own
allocation, and the thing being measured would be entangled with the measuring apparatus.
With tagged ints, the only allocation a Mite program causes is the one the Mite heap records.

Tagged integers are also order-isomorphic to their values (`mk_int(a) < mk_int(b)` iff
`a < b`), which lets the interpreter compare them without untagging.

## 3. The heap is a cost simulator, not a layout simulator

Objects live in a dict keyed by a stable address. A "copy" flips a space tag and adds the
object's size to a counter; it does not relocate anything.

**What that buys:** every reported number is an exact, deterministic, machine-independent
event count. Two runs on two machines produce identical JSON. A wall-clock number from a
Python interpreter would mostly be measuring CPython.

**What that costs:** no fragmentation (which hurts mark-sweep in reality) and no compaction
or locality benefit (which helps copying collectors in reality). Both effects are absent, so
the mark-sweep and semispace rows are context, not a verdict. The headline comparison —
generational vs generational-with-pretenuring — is unaffected, because those two share
identical machinery.

## 4. Root scanning is exact, and the interpreter is written around that

Every live value is either in the operand stack below `sp` or in the globals array. There are
no values hiding in Python locals across an allocation. This is why the allocating opcodes
look slightly awkward: they read their operands *by stack index*, allocate, and only then
move `sp`.

The interpreter loop caches `sp` in a Python local for speed and writes it back to `self.sp`
before anything that can allocate, collect, or raise. Every such point is commented `# root
sync`. Getting this wrong produces a collector that reclaims a live object — which usually
shows up as a plausible wrong answer, not a crash, so `tests/test_gc.py` checks reachability
independently of the collector's own marking.

## 5. Lists are two objects

A Mite list is a header (`{length, backing pointer}`) plus a separately allocated backing
array. `push` doubles the array when full, which allocates a new one and orphans the old.

This is how every real dynamic array works, and it matters here for two reasons: it produces
a realistic pattern of large short-lived allocations, and it creates a case where one
source-level operation allocates twice — which is exactly the case that broke an earlier
version of the collector, because the second allocation could trigger a collection that did
not yet know about the first. That is why registration happens in `on_object_created` (the
instant the object exists) rather than in `post_init` (after its fields are filled).

## 6. Four collectors, one interface

```python
class Collector:
    def alloc(self, kind, size_words, site): ...   # may collect
    def collect(self, reason, space): ...
    def write_barrier(self, obj, value): ...
    def stats(self): ...
```

`MarkSweep` and `SemiSpace` exist as reference points. The experiment is entirely inside
`Generational`, whose `choose_space()` is the single method that differs between the stock
collector, the per-site table, the learned model and the oracle. Nothing else in the pipeline
changes, so no result can be explained by an incidental difference between implementations.

Promotion is age-1: everything surviving one minor collection is promoted. Real collectors
use survivor spaces and a tenuring threshold; modelling those honestly would mean modelling
copy-within-nursery costs, and it was better to leave them out and say so than to model them
badly.

## 7. The remembered set

Old-to-young pointers are caught by a write barrier on every `SET_FIELD`, `SET_INDEX` and
`push`, plus a `post_init` sweep over a pretenured object's initial fields — a pretenured
object is *born* old, so even its constructor stores need the barrier. Missing that produced
a bug where a pretenured record's freshly allocated child was collected out from under it;
`test_pretenured_objects_register_in_the_remembered_set` is the regression test.

A minor collection scans the remembered set in full and charges it as tracing work. There is
no card table; the set holds whole objects.

## 8. Lifetime ground truth

`LifetimeTracer` is a collector with a 1 KB budget that writes down what it killed. Reusing
the collector interface is not a trick — "find everything unreachable right now" *is* a
collection.

Lifetime is measured in bytes allocated since birth. That unit is a property of the program:
it does not change with heap size, collector, or machine, which is what lets a lifetime
recorded under the tracer be applied under a completely different collector.

Death is detected at the next checkpoint, so lifetimes are over-estimated by up to 1 KB.
Hertz & Berger's Merlin algorithm reconstructs exact death times from timestamps at a
fraction of the cost and is the right long-term fix.

## 9. The policy family

The first version of the experiment defined the oracle as "pretenure exactly the objects that
survive one nursery" and produced a nonsense result: on two workloads the perfect collector
lost to the trivial policy of pretenuring everything.

The target was wrong, not the oracle. An object that outlives the nursery by a hair gets
promoted and then dies — pretenuring it moves garbage into a space only a full trace can
reclaim. So pretenuring is a family:

> pretenure iff the object will live at least **K** nurseries

`K = 0` is always-pretenure, `K = ∞` is the stock collector, and both are members. The oracle
sweeps `K` with full hindsight and takes the best, making it a genuine upper bound on
lifetime-based pretenuring. The learner's target is "live at least `K*` nurseries" with `K*`
chosen on the profiling run only.

One consequence worth stating: the oracle bounds a policy *shape*, not the problem. On
`binarytrees` the learned model beats it, because the learned model is not confined to a
threshold on true lifetime.

## 10. Features: two rules

**Mutator-visible state only.** Nursery occupancy, bytes since the last GC and heap pressure
are all obvious features and all forbidden, because they differ between collector
configurations. A model trained on the tracing run's heap state and served under a different
heap state would degrade for reasons indistinguishable from the model being bad. Features are
restricted to: allocation site, enclosing function, caller, object size, call depth, and how
many objects this site has already produced.

**One definition, two call sites.** `alloc_context()` captures six integers; `tokens_from_
context()` expands them into sparse string tokens. The tracer stores contexts and the trainer
expands them later; the collector calls both in sequence at run time.
`test_online_and_offline_features_are_identical` asserts the two paths produce identical token
sequences, allocation for allocation. Training/serving skew is the classic way to build a
model that looks excellent offline and does nothing in production, and it fails silently.

## 11. The model

Feature-hashed logistic regression trained with plain SGD, ~120 lines, no dependencies.

Hashing uses CRC32 rather than Python's `hash()`, which is randomised per process — a model
trained in one process would silently score garbage in the next.

Predictions are memoised by feature vector. Across the whole suite there are only 43–138
distinct feature vectors per program against tens of thousands of allocations, so at run time
the "model" is a dict lookup. That is the difference between a policy you could ship and a
thought experiment.

A linear model was chosen deliberately over anything stronger: if a linear model on hashed
site features already recovers most of the achievable win, that is the interesting result,
and reaching for gradient boosting first would have hidden it.

## 12. The cost model

```
GC work = bytes_traced + 2.0 × bytes_copied + 0.1 × bytes_swept
```

Tracing reads a header and its pointer slots. Copying reads *and* writes *and* fixes up
references, so it is charged double. Sweeping walks headers in address order without following
pointers, so it is charged a tenth.

The constants are arguable. Two things make that survivable: `results/results.json` keeps the
raw component counters so anyone can re-weight without re-running, and the headline comparison
is between collectors that differ only in object placement, so their ordering is independent
of the weights.

## 13. Train/test separation

Each benchmark is compiled twice from one source with different injected constants (`seed`,
`scale`) — different input size, different data, different branch outcomes. The prologue that
injects them contains no allocation sites, so site ids are identical between variants, which
is precisely what makes a profile gathered on one applicable to the other.

The predictor is trained on the profiling variant. `K*` and the decision threshold are chosen
on the profiling variant. Every number in the results table comes from the evaluation variant.

The oracle is allowed hindsight on both counts, and that is labelled wherever it appears.
