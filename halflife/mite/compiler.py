"""Mite AST -> bytecode.

The compiler's most important job for this project is *not* code quality. It is
assigning every allocation in the program a stable **allocation site id**, and
recording what we statically know about that site (function, line, shape, kind).
Those ids are the feature the lifetime predictor keys off, and the reason the
whole experiment is possible: two `record { ... }` expressions on different lines
are different sites even if they build identical objects.
"""

from . import ast_nodes as A
from .opcodes import Op

#: (name, arity, allocates)
NATIVES = [
    ("print", 1, False),
    ("len", 1, False),
    ("push", 2, True),
    ("str", 1, True),
    ("assert", 2, False),
    ("hash", 1, False),
]
NATIVE_INDEX = {name: i for i, (name, _, _) in enumerate(NATIVES)}


class CompileError(Exception):
    pass


class AllocSite:
    """Static description of one allocation point in the program."""

    __slots__ = ("id", "kind", "function", "line", "shape", "detail")

    def __init__(self, id_, kind, function, line, shape=None, detail=""):
        self.id = id_
        self.kind = kind          # record | list | string | concat | native
        self.function = function
        self.line = line
        self.shape = shape        # tuple of field names for records
        self.detail = detail

    def label(self):
        base = "%s:%d %s" % (self.function, self.line, self.kind)
        if self.shape:
            base += "{%s}" % ",".join(self.shape)
        elif self.detail:
            base += "(%s)" % self.detail
        return base

    def to_json(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "function": self.function,
            "line": self.line,
            "shape": list(self.shape) if self.shape else None,
            "detail": self.detail,
            "label": self.label(),
        }


class Function:
    __slots__ = ("name", "arity", "nlocals", "code", "lines", "index")

    def __init__(self, name, arity, index):
        self.name = name
        self.arity = arity
        self.index = index
        self.nlocals = arity
        self.code = []
        self.lines = []

    def emit(self, line, op, *args):
        self.code.append((op,) + args)
        self.lines.append(line)
        return len(self.code) - 1

    def __repr__(self):
        return "<Function %s/%d %d instrs>" % (self.name, self.arity, len(self.code))


class Module:
    """A fully compiled Mite program."""

    def __init__(self):
        self.functions = []
        self.func_index = {}
        self.globals = []
        self.global_index = {}
        self.int_consts = []
        self.int_index = {}
        self.strings = []
        self.string_index = {}
        self.names = []
        self.name_index = {}
        self.shapes = []
        self.shape_index = {}
        self.sites = []
        self.entry = 0

    # --- interning ---------------------------------------------------------
    def intern_int(self, v):
        if v not in self.int_index:
            self.int_index[v] = len(self.int_consts)
            self.int_consts.append(v)
        return self.int_index[v]

    def intern_string(self, s):
        if s not in self.string_index:
            self.string_index[s] = len(self.strings)
            self.strings.append(s)
        return self.string_index[s]

    def intern_name(self, s):
        if s not in self.name_index:
            self.name_index[s] = len(self.names)
            self.names.append(s)
        return self.name_index[s]

    def intern_shape(self, keys):
        """Record shapes are interned, exactly like hidden classes in a real VM.

        A shape is *not* a heap object: it lives in the module, so a record with
        four fields costs a header plus four slots, not four key strings. This
        keeps the measured heap traffic representative of a real runtime.
        """
        key = tuple(keys)
        if key not in self.shape_index:
            self.shape_index[key] = len(self.shapes)
            self.shapes.append(key)
        return self.shape_index[key]

    def new_site(self, kind, function, line, shape=None, detail=""):
        site = AllocSite(len(self.sites), kind, function, line, shape, detail)
        self.sites.append(site)
        return site.id

    def disassemble(self, func=None):
        out = []
        funcs = self.functions if func is None else [self.functions[self.func_index[func]]]
        for f in funcs:
            out.append("fun %s/%d  (locals=%d)" % (f.name, f.arity, f.nlocals))
            for i, instr in enumerate(f.code):
                op = instr[0]
                args = " ".join(str(a) for a in instr[1:])
                out.append("  %4d  %-18s %s" % (i, Op(op).name, args))
        return "\n".join(out)


class _Scope:
    def __init__(self, parent=None):
        self.parent = parent
        self.vars = {}

    def lookup(self, name):
        s = self
        while s is not None:
            if name in s.vars:
                return s.vars[name]
            s = s.parent
        return None


class Compiler:
    def __init__(self, filename="<mite>"):
        self.module = Module()
        self.filename = filename
        self.fn = None
        self.scope = None
        self.loop_stack = []
        self.top_level = False

    # --- helpers -----------------------------------------------------------
    def err(self, node, msg):
        raise CompileError("%s:%d: %s" % (self.filename, getattr(node, "line", 0), msg))

    def declare_local(self, name):
        slot = self.fn.nlocals
        self.fn.nlocals += 1
        self.scope.vars[name] = slot
        return slot

    def push_scope(self):
        self.scope = _Scope(self.scope)

    def pop_scope(self):
        self.scope = self.scope.parent

    # --- entry -------------------------------------------------------------
    def compile(self, program):
        m = self.module

        main = Function("@main", 0, 0)
        m.functions.append(main)
        m.func_index["@main"] = 0

        # Pre-declare all functions so calls can appear before definitions.
        for decl in program.functions:
            if decl.name in m.func_index:
                self.err(decl, "duplicate function %r" % decl.name)
            if decl.name in NATIVE_INDEX:
                self.err(decl, "%r shadows a builtin" % decl.name)
            f = Function(decl.name, len(decl.params), len(m.functions))
            m.func_index[decl.name] = f.index
            m.functions.append(f)

        # Top-level statements become @main; top-level `let`s become globals.
        self.fn = main
        self.push_scope()
        self.top_level = True
        for stmt in program.main:
            self.statement(stmt)
        self.pop_scope()
        main.emit(0, Op.HALT)
        self.top_level = False

        for decl in program.functions:
            self.compile_function(decl)

        return m

    def compile_function(self, decl):
        self.fn = self.module.functions[self.module.func_index[decl.name]]
        self.push_scope()
        for i, p in enumerate(decl.params):
            if p in self.scope.vars:
                self.err(decl, "duplicate parameter %r" % p)
            self.scope.vars[p] = i
        for stmt in decl.body.statements:
            self.statement(stmt)
        self.pop_scope()
        self.fn.emit(decl.line, Op.NIL)
        self.fn.emit(decl.line, Op.RET)

    # --- statements --------------------------------------------------------
    def statement(self, node):
        cls = type(node).__name__
        getattr(self, "st_" + cls)(node)

    def st_Block(self, node):
        self.push_scope()
        was_top = self.top_level
        self.top_level = False
        for s in node.statements:
            self.statement(s)
        self.top_level = was_top
        self.pop_scope()

    def st_Let(self, node):
        self.expr(node.value)
        if self.top_level and self.fn.index == 0:
            m = self.module
            if node.name in m.global_index:
                self.err(node, "duplicate global %r" % node.name)
            idx = len(m.globals)
            m.global_index[node.name] = idx
            m.globals.append(node.name)
            self.fn.emit(node.line, Op.STORE_GLOBAL, idx)
        else:
            if node.name in self.scope.vars:
                self.err(node, "%r already declared in this scope" % node.name)
            slot = self.declare_local(node.name)
            self.fn.emit(node.line, Op.STORE_LOCAL, slot)

    def st_Assign(self, node):
        t = node.target
        if isinstance(t, A.Name):
            self.expr(node.value)
            slot = self.scope.lookup(t.name)
            if slot is not None:
                self.fn.emit(node.line, Op.STORE_LOCAL, slot)
            elif t.name in self.module.global_index:
                self.fn.emit(node.line, Op.STORE_GLOBAL, self.module.global_index[t.name])
            else:
                self.err(node, "assignment to undeclared variable %r" % t.name)
        elif isinstance(t, A.FieldAccess):
            self.expr(t.obj)
            self.expr(node.value)
            self.fn.emit(node.line, Op.SET_FIELD, self.module.intern_name(t.name))
        elif isinstance(t, A.Index):
            self.expr(t.obj)
            self.expr(t.index)
            self.expr(node.value)
            self.fn.emit(node.line, Op.SET_INDEX)
        else:
            self.err(node, "invalid assignment target")

    def st_ExprStmt(self, node):
        self.expr(node.expr)
        self.fn.emit(node.line, Op.POP)

    def st_Return(self, node):
        if self.fn.index == 0:
            self.err(node, "`return` outside of a function")
        if node.value is None:
            self.fn.emit(node.line, Op.NIL)
        else:
            self.expr(node.value)
        self.fn.emit(node.line, Op.RET)

    def st_If(self, node):
        self.expr(node.cond)
        jf = self.fn.emit(node.line, Op.JUMP_IF_FALSE, -1)
        self.statement(node.then_body)
        if node.else_body is not None:
            jend = self.fn.emit(node.line, Op.JUMP, -1)
            self.patch(jf)
            self.statement(node.else_body)
            self.patch(jend)
        else:
            self.patch(jf)

    def st_While(self, node):
        top = len(self.fn.code)
        self.expr(node.cond)
        jf = self.fn.emit(node.line, Op.JUMP_IF_FALSE, -1)
        self.loop_stack.append({"top": top, "breaks": [], "continues": []})
        self.statement(node.body)
        frame = self.loop_stack.pop()
        for c in frame["continues"]:
            self.patch(c, top)
        self.fn.emit(node.line, Op.JUMP, top)
        self.patch(jf)
        for b in frame["breaks"]:
            self.patch(b)

    def st_Break(self, node):
        if not self.loop_stack:
            self.err(node, "break outside loop")
        self.loop_stack[-1]["breaks"].append(self.fn.emit(node.line, Op.JUMP, -1))

    def st_Continue(self, node):
        if not self.loop_stack:
            self.err(node, "continue outside loop")
        self.loop_stack[-1]["continues"].append(self.fn.emit(node.line, Op.JUMP, -1))

    def patch(self, idx, target=None):
        if target is None:
            target = len(self.fn.code)
        instr = self.fn.code[idx]
        self.fn.code[idx] = (instr[0], target) + instr[2:]

    # --- expressions -------------------------------------------------------
    def expr(self, node):
        getattr(self, "ex_" + type(node).__name__)(node)

    def ex_IntLit(self, node):
        self.fn.emit(node.line, Op.CONST, self.module.intern_int(node.value))

    def ex_StrLit(self, node):
        k = self.module.intern_string(node.value)
        site = self.module.new_site(
            "string", self.fn.name, node.line, detail=repr(node.value[:16])
        )
        self.fn.emit(node.line, Op.NEW_STRING, k, site)

    def ex_BoolLit(self, node):
        self.fn.emit(node.line, Op.TRUE if node.value else Op.FALSE)

    def ex_NilLit(self, node):
        self.fn.emit(node.line, Op.NIL)

    def ex_Name(self, node):
        slot = self.scope.lookup(node.name)
        if slot is not None:
            self.fn.emit(node.line, Op.LOAD_LOCAL, slot)
            return
        if node.name in self.module.global_index:
            self.fn.emit(node.line, Op.LOAD_GLOBAL, self.module.global_index[node.name])
            return
        self.err(node, "undefined variable %r" % node.name)

    _BINOPS = {
        "-": Op.SUB, "*": Op.MUL, "/": Op.DIV, "%": Op.MOD,
        "==": Op.EQ, "!=": Op.NE, "<": Op.LT, "<=": Op.LE,
        ">": Op.GT, ">=": Op.GE,
    }

    def ex_Binary(self, node):
        self.expr(node.left)
        self.expr(node.right)
        if node.op == "+":
            # `+` is polymorphic: int addition, or string concatenation, which
            # allocates. The site is reserved at compile time either way.
            site = self.module.new_site("concat", self.fn.name, node.line, detail="+")
            self.fn.emit(node.line, Op.ADD, site)
        else:
            self.fn.emit(node.line, self._BINOPS[node.op])

    def ex_Unary(self, node):
        self.expr(node.operand)
        self.fn.emit(node.line, Op.NEG if node.op == "-" else Op.NOT)

    def ex_Logical(self, node):
        self.expr(node.left)
        op = Op.JUMP_IF_FALSE_KEEP if node.op == "and" else Op.JUMP_IF_TRUE_KEEP
        j = self.fn.emit(node.line, op, -1)
        self.fn.emit(node.line, Op.POP)
        self.expr(node.right)
        self.patch(j)

    def ex_Call(self, node):
        if not isinstance(node.callee, A.Name):
            self.err(node, "only direct calls to named functions are supported")
        name = node.callee.name
        for a in node.args:
            self.expr(a)
        if name in NATIVE_INDEX:
            idx = NATIVE_INDEX[name]
            _, arity, allocates = NATIVES[idx]
            if len(node.args) != arity:
                self.err(node, "%s() takes %d argument(s)" % (name, arity))
            site = -1
            if allocates:
                site = self.module.new_site(
                    "native", self.fn.name, node.line, detail=name
                )
            self.fn.emit(node.line, Op.NATIVE, idx, arity, site)
            return
        if name not in self.module.func_index:
            self.err(node, "undefined function %r" % name)
        f = self.module.functions[self.module.func_index[name]]
        if len(node.args) != f.arity:
            self.err(node, "%s() takes %d argument(s), got %d" % (name, f.arity, len(node.args)))
        self.fn.emit(node.line, Op.CALL, f.index, f.arity)

    def ex_RecordLit(self, node):
        for v in node.values:
            self.expr(v)
        shape_idx = self.module.intern_shape(node.keys)
        site = self.module.new_site(
            "record", self.fn.name, node.line, shape=tuple(node.keys)
        )
        self.fn.emit(node.line, Op.NEW_RECORD, shape_idx, site)

    def ex_ListLit(self, node):
        for e in node.elements:
            self.expr(e)
        site = self.module.new_site(
            "list", self.fn.name, node.line, detail="n=%d" % len(node.elements)
        )
        self.fn.emit(node.line, Op.NEW_LIST, len(node.elements), site)

    def ex_FieldAccess(self, node):
        self.expr(node.obj)
        self.fn.emit(node.line, Op.GET_FIELD, self.module.intern_name(node.name))

    def ex_Index(self, node):
        self.expr(node.obj)
        self.expr(node.index)
        self.fn.emit(node.line, Op.GET_INDEX)


def compile_source(src, filename="<mite>"):
    from .parser import parse

    return Compiler(filename).compile(parse(src, filename))
