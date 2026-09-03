"""Tagged value representation for the Mite VM.

Every Mite value is a single Python ``int``, the way a real VM packs a value into
a machine word. The low three bits are the tag:

    ``0b000``  heap reference   -- address is ``v >> 3`` (address 0 is reserved,
                                  so a valid reference is never numerically 0)
    ``0b001``  small integer    -- value is ``v >> 3`` (arithmetic shift, so
                                  negatives round-trip correctly)
    ``0b010``  constant         -- nil / false / true, distinguished by ``v >> 3``

Using ints rather than Python objects keeps the mutator allocation-free: the
*only* thing that ever allocates during a Mite program is the Mite heap, which
is exactly what we are trying to measure. If values were Python objects, CPython's
own allocator would be entangled with the experiment.
"""

TAG_BITS = 3
TAG_MASK = 0b111

TAG_REF = 0b000
TAG_INT = 0b001
TAG_CONST = 0b010

NIL = TAG_CONST | (0 << TAG_BITS)
FALSE = TAG_CONST | (1 << TAG_BITS)
TRUE = TAG_CONST | (2 << TAG_BITS)


def mk_int(n):
    return (n << TAG_BITS) | TAG_INT


def as_int(v):
    return v >> TAG_BITS


def is_int(v):
    return (v & TAG_MASK) == TAG_INT


def mk_ref(addr):
    if addr <= 0:
        raise ValueError("heap address 0 is reserved")
    return addr << TAG_BITS


def as_ref(v):
    return v >> TAG_BITS


def is_ref(v):
    return (v & TAG_MASK) == TAG_REF


def mk_bool(b):
    return TRUE if b else FALSE


def truthy(v):
    """Mite truthiness: only ``false`` and ``nil`` are falsey."""
    return v != FALSE and v != NIL


def type_name(v, heap=None):
    if is_int(v):
        return "int"
    if v == NIL:
        return "nil"
    if v in (TRUE, FALSE):
        return "bool"
    if is_ref(v):
        if heap is None:
            return "ref"
        return heap.objects[as_ref(v)].kind_name()
    return "?"
