"""Bytecode instruction set for Mite.

Instructions are tuples ``(Op, *operands)``. A flat tuple encoding (rather than a
packed byte array) is deliberate: this is a *simulator*, and every metric we
report is a counted event rather than a wall-clock measurement, so decode speed
does not affect a single number in the results table.
"""

from enum import IntEnum


class Op(IntEnum):
    # --- stack / constants -------------------------------------------------
    CONST = 0           # (k)      push constant pool entry k (an int literal)
    NIL = 1
    TRUE = 2
    FALSE = 3
    POP = 4
    DUP = 5

    # --- variables ---------------------------------------------------------
    LOAD_LOCAL = 10     # (slot)
    STORE_LOCAL = 11    # (slot)
    LOAD_GLOBAL = 12    # (idx)
    STORE_GLOBAL = 13   # (idx)

    # --- arithmetic / comparison ------------------------------------------
    ADD = 20
    SUB = 21
    MUL = 22
    DIV = 23
    MOD = 24
    NEG = 25
    EQ = 26
    NE = 27
    LT = 28
    LE = 29
    GT = 30
    GE = 31
    NOT = 32

    # --- control flow ------------------------------------------------------
    JUMP = 40             # (target)
    JUMP_IF_FALSE = 41    # (target)   pops
    JUMP_IF_FALSE_KEEP = 42  # (target) leaves value on stack (for `and`)
    JUMP_IF_TRUE_KEEP = 43   # (target) leaves value on stack (for `or`)
    CALL = 44             # (func_idx, argc)
    RET = 45
    NATIVE = 46           # (native_idx, argc)

    # --- allocation --------------------------------------------------------
    NEW_RECORD = 50       # (shape_idx, site_id)  pops nfields values
    NEW_LIST = 51         # (count, site_id)      pops count values
    NEW_STRING = 52       # (const_idx, site_id)
    CONCAT = 53           # (site_id)             pops 2, allocates

    # --- heap access -------------------------------------------------------
    GET_FIELD = 60        # (name_idx)
    SET_FIELD = 61        # (name_idx)
    GET_INDEX = 62
    SET_INDEX = 63

    HALT = 90


#: Instructions that can trigger a heap allocation (and therefore a collection).
#: ``ADD`` is in here because ``+`` is polymorphic: on strings it concatenates,
#: which allocates. ``NATIVE`` is in here because ``push``/``str`` allocate.
ALLOCATING_OPS = frozenset(
    {Op.NEW_RECORD, Op.NEW_LIST, Op.NEW_STRING, Op.ADD, Op.NATIVE}
)
