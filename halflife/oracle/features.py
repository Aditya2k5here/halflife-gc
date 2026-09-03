"""Features for the lifetime predictor.

There is exactly one place features are defined, and both the offline trainer and
the online collector reach it. That is not tidiness for its own sake: a lifetime
predictor that sees different features at training time and serving time is the
classic way to build a model that looks excellent in a notebook and does nothing
in production. Sharing the code makes training/serving skew impossible rather
than unlikely.

The split between :func:`alloc_context` (six integers, captured at allocation
time) and :func:`tokens_from_context` (the sparse string features) exists so the
tracer can record hundreds of thousands of allocations compactly and regenerate
the exact same tokens later.

Two rules constrain what may go in here:

1. **Mutator-visible state only.** Nursery occupancy, time since the last GC and
   heap pressure are all tempting and all forbidden, because they differ between
   collector configurations. A model trained on the tracing run's heap state
   would be served under a different heap state, and the resulting drop would be
   indistinguishable from the model simply being bad.
2. **Cheap to compute.** These are extracted on *every allocation* in the fast
   path. A feature that requires walking the call stack is not a feature, it is
   a tax.
"""

#: field order of the context tuple
CONTEXT_FIELDS = ("site", "size_words", "func", "caller", "depth", "n_prev")


def _log_bucket(n):
    """0 -> 0, 1 -> 1, 2..3 -> 2, 4..7 -> 3, ... Counts vary over orders of magnitude."""
    b = 0
    while n > 0:
        n >>= 1
        b += 1
    return b


def alloc_context(site_id, size_words, vm):
    """Capture the (cheap, collector-independent) state of one allocation."""
    return (
        site_id,
        size_words,
        vm.current_func_index,
        vm.caller_func_index,
        vm.call_depth,
        vm.site_alloc_count[site_id],
    )


def tokens_from_context(ctx, module):
    site_id, size_words, fn, caller, depth, n_prev = ctx
    kind = module.sites[site_id].kind
    size_b = _log_bucket(size_words)
    count_b = _log_bucket(n_prev)
    depth_b = depth if depth < 8 else 8 + _log_bucket(depth - 7)
    return (
        "bias",
        "site=%d" % site_id,
        "kind=%s" % kind,
        "fn=%d" % fn,
        "caller=%d" % caller,
        "size=%d" % size_b,
        "depth=%d" % depth_b,
        "count=%d" % count_b,
        # Conjunctions matter: the same site behaves differently depending on who
        # reached it. A node allocated by `build` becomes part of a structure
        # that lives; the identical expression reached from a scratch helper does
        # not. A model without these can only learn a per-site average.
        "site^caller=%d^%d" % (site_id, caller),
        "site^depth=%d^%d" % (site_id, depth_b),
        "site^count=%d^%d" % (site_id, count_b),
        "kind^depth=%s^%d" % (kind, depth_b),
    )


def alloc_tokens(site_id, size_words, vm):
    """Online path: context and tokens in one step."""
    return tokens_from_context(alloc_context(site_id, size_words, vm), vm.module)


#: Tokens that name something specific to one program. ``site=7`` means nothing
#: outside the program it was compiled from -- and worse than nothing, because
#: two different programs' site 7 hash to the same feature and interfere.
PROGRAM_SPECIFIC = ("site", "fn=", "caller=")


def transferable(tokens):
    """Drop program-specific tokens, leaving only features that mean the same
    thing in any program: allocation kind, object size, call depth, and how many
    objects this site has already produced.

    This exists to answer a separate question from the main experiment: is there
    a *program-independent* signal in allocation behaviour, or does every
    program need its own profile? The answer turns out to be interesting, and it
    is reported as its own row in the results table.
    """
    return tuple(t for t in tokens if not t.startswith(PROGRAM_SPECIFIC))
