"""The Mite bytecode interpreter.

The interpreter is ordinary; what matters here is that it is **precise** and
**deterministic**.

*Precise* means the collector always has an exact root set. Every live value is
either in the operand stack below ``sp`` or in the globals array -- there are no
values hiding in Python locals across an allocation. Allocation therefore always
happens *before* operands are popped, which is why the ``NEW_*`` handlers look
slightly awkward: they read their operands by index, allocate, and only then
move ``sp``.

*Deterministic* means the same program produces byte-for-byte the same
allocation sequence on every run, regardless of which collector is installed.
That property is what makes the whole experiment sound: a lifetime recorded for
allocation #12345 during the tracing run describes the same object that
allocation #12345 produces during the evaluation run. ``tests/test_determinism.py``
asserts it rather than assuming it.
"""

from ..values import (
    NIL, TRUE, FALSE, mk_int, as_int, is_int, mk_ref, as_ref, is_ref, truthy,
)
from ..mite.opcodes import Op
from ..mite.compiler import NATIVES
from .heap import (
    Heap, KIND_RECORD, KIND_LIST, KIND_ARRAY, KIND_STRING,
    array_words, string_words,
)

STACK_MAX = 1 << 16
MAX_FRAMES = 4096


class MiteError(Exception):
    pass


class Frame(object):
    __slots__ = ("func", "ip", "base")

    def __init__(self, func, ip, base):
        self.func = func
        self.ip = ip
        self.base = base


class VM(object):
    def __init__(self, module, collector, max_instructions=200_000_000,
                 alloc_observer=None, capture_output=True,
                 record_heap_events=False, max_events=60000):
        self.module = module
        self.gc = collector
        self.heap = Heap(record_events=record_heap_events, max_events=max_events)
        self.max_instructions = max_instructions
        self.alloc_observer = alloc_observer
        self.capture_output = capture_output

        self.stack = [NIL] * STACK_MAX
        self.sp = 0
        self.globals = [NIL] * max(1, len(module.globals))
        self.frames = []
        self.func = None
        self.ip = 0
        self.base = 0

        self.instructions = 0
        self.output = []
        self.site_alloc_count = [0] * max(1, len(module.sites))
        self._field_cache = {}

        collector.attach(self)

    # ------------------------------------------------------------------ roots
    def iter_roots(self):
        """Exact root set: the live prefix of the operand stack, plus globals."""
        stack = self.stack
        for i in range(self.sp):
            yield stack[i]
        for v in self.globals:
            yield v

    # -------------------------------------------------------- context for ML
    @property
    def current_func_index(self):
        return self.func.index if self.func is not None else -1

    @property
    def caller_func_index(self):
        return self.frames[-1].func.index if self.frames else -1

    @property
    def call_depth(self):
        return len(self.frames)

    # ------------------------------------------------------------ allocation
    def _after_alloc(self, obj):
        self.gc.post_init(obj)
        if self.alloc_observer is not None:
            self.alloc_observer(obj, self)
        self.site_alloc_count[obj.site] += 1

    def alloc_string(self, text, site):
        obj = self.gc.alloc(KIND_STRING, string_words(len(text)), site)
        obj.text = text
        obj.length = len(text)
        self._after_alloc(obj)
        return obj

    def alloc_array(self, capacity, site):
        obj = self.gc.alloc(KIND_ARRAY, array_words(capacity), site)
        obj.fields = [NIL] * capacity
        obj.length = capacity
        self._after_alloc(obj)
        return obj

    # ------------------------------------------------------------------- run
    def run(self):
        """The interpreter loop.

        Three deliberate performance concessions, because this loop executes
        tens of millions of instructions per experiment and everything here is a
        simulation whose *results* are event counts, not timings:

        * opcode numbers and hot globals are hoisted into locals -- in CPython a
          local is an array index while ``Op.LOAD_LOCAL`` is a global lookup plus
          an attribute lookup, paid on every instruction;
        * integer constants are pre-tagged once, so ``CONST`` is a list index;
        * ``sp`` lives in a local and is written back to ``self.sp`` *before*
          anything that can allocate, collect, or raise. That write-back is not
          optional: it is what keeps the collector's root set exact. Every place
          it happens is marked.
        """
        m = self.module
        consts = [mk_int(v) for v in m.int_consts]
        shapes = m.shapes
        functions = m.functions
        stack = self.stack
        gc = self.gc
        objects = self.heap.objects

        LOAD_LOCAL, STORE_LOCAL, CONST = Op.LOAD_LOCAL, Op.STORE_LOCAL, Op.CONST
        LOAD_GLOBAL, STORE_GLOBAL = Op.LOAD_GLOBAL, Op.STORE_GLOBAL
        ADD, SUB, MUL, DIV, MOD, NEG = Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD, Op.NEG
        EQ, NE, LT, LE, GT, GE, NOT = Op.EQ, Op.NE, Op.LT, Op.LE, Op.GT, Op.GE, Op.NOT
        JUMP, JUMP_IF_FALSE = Op.JUMP, Op.JUMP_IF_FALSE
        JF_KEEP, JT_KEEP = Op.JUMP_IF_FALSE_KEEP, Op.JUMP_IF_TRUE_KEEP
        CALL, RET, NATIVE = Op.CALL, Op.RET, Op.NATIVE
        NEW_RECORD, NEW_LIST, NEW_STRING = Op.NEW_RECORD, Op.NEW_LIST, Op.NEW_STRING
        GET_FIELD, SET_FIELD = Op.GET_FIELD, Op.SET_FIELD
        GET_INDEX, SET_INDEX = Op.GET_INDEX, Op.SET_INDEX
        POP, DUP, NILOP, TRUEOP, FALSEOP, HALT = (
            Op.POP, Op.DUP, Op.NIL, Op.TRUE, Op.FALSE, Op.HALT
        )

        self.func = func = m.functions[m.entry]
        code = func.code
        ip = 0
        base = 0
        self.base = 0
        sp = func.nlocals
        for i in range(sp):
            stack[i] = NIL
        count = 0
        budget = self.max_instructions
        globals_ = self.globals

        while True:
            count += 1
            if count > budget:
                self.sp = sp
                raise MiteError("instruction budget exhausted (possible infinite loop)")

            instr = code[ip]
            ip += 1
            op = instr[0]

            if op == LOAD_LOCAL:
                stack[sp] = stack[base + instr[1]]
                sp += 1
            elif op == CONST:
                stack[sp] = consts[instr[1]]
                sp += 1
            elif op == STORE_LOCAL:
                sp -= 1
                stack[base + instr[1]] = stack[sp]

            # --- arithmetic. Tag manipulation is inlined: tag 1 is a small int,
            # so `v >> 3` is its value and `(n << 3) | 1` re-tags it.
            elif op == ADD:
                b = stack[sp - 1]
                a = stack[sp - 2]
                if (a & 7) == 1 and (b & 7) == 1:
                    stack[sp - 2] = (((a >> 3) + (b >> 3)) << 3) | 1
                    sp -= 1
                else:
                    self.sp = sp                      # root sync: may allocate
                    stack[sp - 2] = self._concat(a, b, instr[1])
                    sp -= 1
            elif op == SUB:
                sp -= 1
                a = stack[sp - 1]
                b = stack[sp]
                if (a & 7) == 1 and (b & 7) == 1:
                    stack[sp - 1] = (((a >> 3) - (b >> 3)) << 3) | 1
                else:
                    raise MiteError("`-` expects two integers")
            elif op == MUL:
                sp -= 1
                a = stack[sp - 1]
                b = stack[sp]
                if (a & 7) == 1 and (b & 7) == 1:
                    stack[sp - 1] = (((a >> 3) * (b >> 3)) << 3) | 1
                else:
                    raise MiteError("`*` expects two integers")
            elif op == LT:
                sp -= 1
                a = stack[sp - 1]
                b = stack[sp]
                if (a & 7) == 1 and (b & 7) == 1:
                    stack[sp - 1] = TRUE if a < b else FALSE
                else:
                    raise MiteError("`<` expects two integers")
            elif op == MOD:
                sp -= 1
                a = self._int(stack[sp - 1])
                b = self._int(stack[sp])
                stack[sp - 1] = mk_int(a - self._trunc_div(a, b) * b)
            elif op == DIV:
                sp -= 1
                stack[sp - 1] = mk_int(
                    self._trunc_div(self._int(stack[sp - 1]), self._int(stack[sp]))
                )
            elif op == NEG:
                stack[sp - 1] = mk_int(-self._int(stack[sp - 1]))

            # --- remaining comparisons. Tagged ints are order-isomorphic to
            # their values, so `<` on the raw words is correct for ints.
            elif op == LE:
                sp -= 1
                a, b = stack[sp - 1], stack[sp]
                if (a & 7) == 1 and (b & 7) == 1:
                    stack[sp - 1] = TRUE if a <= b else FALSE
                else:
                    raise MiteError("`<=` expects two integers")
            elif op == GT:
                sp -= 1
                a, b = stack[sp - 1], stack[sp]
                if (a & 7) == 1 and (b & 7) == 1:
                    stack[sp - 1] = TRUE if a > b else FALSE
                else:
                    raise MiteError("`>` expects two integers")
            elif op == GE:
                sp -= 1
                a, b = stack[sp - 1], stack[sp]
                if (a & 7) == 1 and (b & 7) == 1:
                    stack[sp - 1] = TRUE if a >= b else FALSE
                else:
                    raise MiteError("`>=` expects two integers")
            elif op == EQ:
                sp -= 1
                a, b = stack[sp - 1], stack[sp]
                stack[sp - 1] = TRUE if (a == b or self._value_eq(a, b)) else FALSE
            elif op == NE:
                sp -= 1
                a, b = stack[sp - 1], stack[sp]
                stack[sp - 1] = FALSE if (a == b or self._value_eq(a, b)) else TRUE
            elif op == NOT:
                v = stack[sp - 1]
                stack[sp - 1] = TRUE if (v == FALSE or v == NIL) else FALSE

            # --- control flow ------------------------------------------------
            elif op == JUMP:
                ip = instr[1]
            elif op == JUMP_IF_FALSE:
                sp -= 1
                v = stack[sp]
                if v == FALSE or v == NIL:
                    ip = instr[1]
            elif op == JF_KEEP:
                v = stack[sp - 1]
                if v == FALSE or v == NIL:
                    ip = instr[1]
            elif op == JT_KEEP:
                v = stack[sp - 1]
                if v != FALSE and v != NIL:
                    ip = instr[1]
            elif op == CALL:
                callee = functions[instr[1]]
                argc = instr[2]
                new_base = sp - argc
                if len(self.frames) >= MAX_FRAMES:
                    self.sp = sp
                    raise MiteError("call stack overflow in %s" % callee.name)
                self.frames.append(Frame(func, ip, base))
                for i in range(argc, callee.nlocals):
                    stack[new_base + i] = NIL
                self.func = func = callee
                base = self.base = new_base
                sp = new_base + callee.nlocals
                ip = 0
                code = callee.code
            elif op == RET:
                retval = stack[sp - 1]
                frame = self.frames.pop()
                stack[base] = retval
                sp = base + 1
                self.func = func = frame.func
                ip = frame.ip
                base = self.base = frame.base
                code = func.code
            elif op == POP:
                sp -= 1
            elif op == NILOP:
                stack[sp] = NIL
                sp += 1
            elif op == TRUEOP:
                stack[sp] = TRUE
                sp += 1
            elif op == FALSEOP:
                stack[sp] = FALSE
                sp += 1
            elif op == DUP:
                stack[sp] = stack[sp - 1]
                sp += 1
            elif op == LOAD_GLOBAL:
                stack[sp] = globals_[instr[1]]
                sp += 1
            elif op == STORE_GLOBAL:
                sp -= 1
                globals_[instr[1]] = stack[sp]

            # --- allocation ---------------------------------------------------
            elif op == NEW_RECORD:
                shape = shapes[instr[1]]
                n = len(shape)
                first = sp - n
                self.sp = sp                          # root sync: the field
                obj = gc.alloc(KIND_RECORD, n + 1, instr[2])   # values are roots
                obj.shape = shape
                obj.fields = stack[first:sp]
                sp = first
                self.sp = sp
                self._after_alloc(obj)
                stack[sp] = obj.addr << 3
                sp += 1
            elif op == NEW_LIST:
                n = instr[1]
                site = instr[2]
                first = sp - n
                self.sp = sp                          # root sync
                arr = self.alloc_array(n if n > 0 else 4, site)
                for i in range(n):
                    arr.fields[i] = stack[first + i]
                sp = first
                # keep the backing array reachable across the header allocation
                stack[sp] = arr.addr << 3
                sp += 1
                self.sp = sp                          # root sync
                lst = gc.alloc(KIND_LIST, 3, site)
                lst.fields = [stack[sp - 1]]
                lst.length = n
                sp -= 1
                self.sp = sp
                self._after_alloc(lst)
                stack[sp] = lst.addr << 3
                sp += 1
            elif op == NEW_STRING:
                self.sp = sp                          # root sync
                obj = self.alloc_string(m.strings[instr[1]], instr[2])
                stack[sp] = obj.addr << 3
                sp += 1

            # --- heap access ---------------------------------------------------
            elif op == GET_FIELD:
                ref = stack[sp - 1]
                obj = objects.get(ref >> 3) if (ref & 7) == 0 else None
                if obj is None or obj.kind != KIND_RECORD:
                    self.sp = sp
                    self._deref(ref, KIND_RECORD, "field access")
                stack[sp - 1] = obj.fields[self._slot(obj.shape, instr[1])]
            elif op == SET_FIELD:
                value = stack[sp - 1]
                ref = stack[sp - 2]
                obj = objects.get(ref >> 3) if (ref & 7) == 0 else None
                if obj is None or obj.kind != KIND_RECORD:
                    self.sp = sp
                    self._deref(ref, KIND_RECORD, "field assignment")
                obj.fields[self._slot(obj.shape, instr[1])] = value
                gc.write_barrier(obj, value)
                sp -= 2
            elif op == GET_INDEX:
                idx = stack[sp - 1]
                ref = stack[sp - 2]
                lst = objects.get(ref >> 3) if (ref & 7) == 0 else None
                if lst is None or lst.kind != KIND_LIST or (idx & 7) != 1:
                    self.sp = sp
                    lst = self._deref(ref, KIND_LIST, "indexing")
                    self._int(idx)
                i = idx >> 3
                if i < 0 or i >= lst.length:
                    self.sp = sp
                    raise MiteError(
                        "list index %d out of range (length %d)" % (i, lst.length)
                    )
                sp -= 2
                stack[sp] = objects[lst.fields[0] >> 3].fields[i]
                sp += 1
            elif op == SET_INDEX:
                value = stack[sp - 1]
                idx = stack[sp - 2]
                ref = stack[sp - 3]
                lst = objects.get(ref >> 3) if (ref & 7) == 0 else None
                if lst is None or lst.kind != KIND_LIST or (idx & 7) != 1:
                    self.sp = sp
                    lst = self._deref(ref, KIND_LIST, "index assignment")
                    self._int(idx)
                i = idx >> 3
                if i < 0 or i >= lst.length:
                    self.sp = sp
                    raise MiteError(
                        "list index %d out of range (length %d)" % (i, lst.length)
                    )
                arr = objects[lst.fields[0] >> 3]
                arr.fields[i] = value
                gc.write_barrier(arr, value)
                sp -= 3

            elif op == NATIVE:
                self.sp = sp                          # root sync: may allocate
                self._native(instr[1], instr[2], instr[3])
                sp = self.sp

            elif op == HALT:
                self.sp = sp
                self.ip = ip
                self.instructions = count
                return self

            else:  # pragma: no cover - defensive
                raise MiteError("unhandled opcode %r" % (Op(op).name,))

    # ------------------------------------------------------------- helpers
    def _slot(self, shape, name_idx):
        key = (shape, name_idx)
        slot = self._field_cache.get(key)
        if slot is None:
            name = self.module.names[name_idx]
            if name not in shape:
                raise MiteError(
                    "record {%s} has no field %r" % (",".join(shape), name)
                )
            slot = shape.index(name)
            self._field_cache[key] = slot
        return slot

    def _deref(self, value, kind, what):
        if not is_ref(value):
            raise MiteError("%s on a non-object value" % what)
        obj = self.heap.objects.get(as_ref(value))
        if obj is None:
            raise MiteError("dangling reference during %s -- collector bug" % what)
        if kind is not None and obj.kind != kind:
            raise MiteError("%s on a %s" % (what, obj.kind_name()))
        return obj

    def _int(self, v):
        if not is_int(v):
            raise MiteError("expected an integer")
        return as_int(v)

    @staticmethod
    def _trunc_div(a, b):
        if b == 0:
            raise MiteError("division by zero")
        q = abs(a) // abs(b)
        return q if (a < 0) == (b < 0) else -q

    def _value_eq(self, a, b):
        if a == b:
            return True
        if is_ref(a) and is_ref(b):
            oa = self.heap.objects.get(as_ref(a))
            ob = self.heap.objects.get(as_ref(b))
            if oa is not None and ob is not None:
                if oa.kind == KIND_STRING and ob.kind == KIND_STRING:
                    return oa.text == ob.text
        return False

    def _concat(self, a, b, site):
        sa = self._to_text(a)
        sb = self._to_text(b)
        if sa is None or sb is None:
            raise MiteError("`+` expects two integers or a string on the left")
        obj = self.alloc_string(sa + sb, site)
        return mk_ref(obj.addr)

    def _to_text(self, v):
        if is_int(v):
            return str(as_int(v))
        if v == NIL:
            return "nil"
        if v == TRUE:
            return "true"
        if v == FALSE:
            return "false"
        if is_ref(v):
            obj = self.heap.objects.get(as_ref(v))
            if obj is not None and obj.kind == KIND_STRING:
                return obj.text
        return None

    def _render(self, v):
        text = self._to_text(v)
        if text is not None:
            return text
        obj = self.heap.objects.get(as_ref(v))
        if obj is None:
            return "<dangling>"
        if obj.kind == KIND_LIST:
            arr = self.heap.objects[as_ref(obj.fields[0])]
            return "[" + ", ".join(self._render(arr.fields[i]) for i in range(obj.length)) + "]"
        if obj.kind == KIND_RECORD:
            return "{" + ", ".join(
                "%s: %s" % (k, self._render(obj.fields[i]))
                for i, k in enumerate(obj.shape)
            ) + "}"
        return "<%s>" % obj.kind_name()

    # ------------------------------------------------------------- natives
    def _native(self, idx, argc, site):
        name = NATIVES[idx][0]
        first = self.sp - argc

        if name == "print":
            if self.capture_output:
                self.output.append(self._render(self.stack[first]))
            self.sp = first
            self.stack[self.sp] = NIL
            self.sp += 1
            return

        if name == "len":
            v = self.stack[first]
            obj = self._deref(v, None, "len()")
            if obj.kind == KIND_LIST:
                n = obj.length
            elif obj.kind == KIND_STRING:
                n = obj.length
            elif obj.kind == KIND_RECORD:
                n = len(obj.shape)
            else:
                raise MiteError("len() on a %s" % obj.kind_name())
            self.sp = first
            self.stack[self.sp] = mk_int(n)
            self.sp += 1
            return

        if name == "hash":
            v = self.stack[first]
            if is_int(v):
                h = as_int(v)
            else:
                obj = self._deref(v, KIND_STRING, "hash()")
                h = 2166136261
                for ch in obj.text:            # FNV-1a, so hashing is stable
                    h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
            self.sp = first
            self.stack[self.sp] = mk_int(h)
            self.sp += 1
            return

        if name == "assert":
            cond = self.stack[first]
            if not truthy(cond):
                raise MiteError("assertion failed: %s" % self._render(self.stack[first + 1]))
            self.sp = first
            self.stack[self.sp] = NIL
            self.sp += 1
            return

        if name == "str":
            text = self._to_text(self.stack[first])
            if text is None:
                text = self._render(self.stack[first])
            obj = self.alloc_string(text, site)
            self.sp = first
            self.stack[self.sp] = mk_ref(obj.addr)
            self.sp += 1
            return

        if name == "push":
            # Arguments stay on the stack across the allocation below, because
            # growing the backing array can trigger a collection.
            lst = self._deref(self.stack[first], KIND_LIST, "push()")
            arr = self.heap.objects[as_ref(lst.fields[0])]
            if lst.length >= arr.length:
                new_cap = max(4, arr.length * 2)
                new_arr = self.alloc_array(new_cap, site)
                # re-read: a collection cannot move objects here, but the list
                # header may have been promoted, so fetch through the stack root
                lst = self.heap.objects[as_ref(self.stack[first])]
                old_arr = self.heap.objects[as_ref(lst.fields[0])]
                for i in range(lst.length):
                    new_arr.fields[i] = old_arr.fields[i]
                lst.fields[0] = mk_ref(new_arr.addr)
                self.gc.write_barrier(lst, lst.fields[0])
                arr = new_arr
            value = self.stack[first + 1]
            arr.fields[lst.length] = value
            lst.length += 1
            self.gc.write_barrier(arr, value)
            self.sp = first
            self.stack[self.sp] = NIL
            self.sp += 1
            return

        raise MiteError("unknown native %r" % name)  # pragma: no cover
