"""Mite AST. Plain data classes -- no behaviour, no visitors, no ceremony."""


class Node:
    __slots__ = ("line",)


def _node(name, fields):
    """Build a small AST node class with __slots__ and a readable repr."""

    class _N(Node):
        __slots__ = tuple(fields)

        def __init__(self, *args, line=0):
            if len(args) != len(fields):
                raise TypeError("%s takes %d args" % (name, len(fields)))
            for f, a in zip(fields, args):
                setattr(self, f, a)
            self.line = line

        def __repr__(self):
            inner = ", ".join("%s=%r" % (f, getattr(self, f)) for f in fields)
            return "%s(%s)" % (name, inner)

    _N.__name__ = name
    _N.__qualname__ = name
    _N._fields = tuple(fields)
    return _N


# --- expressions -----------------------------------------------------------
IntLit = _node("IntLit", ["value"])
StrLit = _node("StrLit", ["value"])
BoolLit = _node("BoolLit", ["value"])
NilLit = _node("NilLit", [])
Name = _node("Name", ["name"])
Binary = _node("Binary", ["op", "left", "right"])
Unary = _node("Unary", ["op", "operand"])
Logical = _node("Logical", ["op", "left", "right"])
Call = _node("Call", ["callee", "args"])
RecordLit = _node("RecordLit", ["keys", "values"])
ListLit = _node("ListLit", ["elements"])
FieldAccess = _node("FieldAccess", ["obj", "name"])
Index = _node("Index", ["obj", "index"])

# --- statements ------------------------------------------------------------
Let = _node("Let", ["name", "value"])
Assign = _node("Assign", ["target", "value"])
ExprStmt = _node("ExprStmt", ["expr"])
If = _node("If", ["cond", "then_body", "else_body"])
While = _node("While", ["cond", "body"])
Return = _node("Return", ["value"])
Break = _node("Break", [])
Continue = _node("Continue", [])
Block = _node("Block", ["statements"])
FunctionDecl = _node("FunctionDecl", ["name", "params", "body"])
Program = _node("Program", ["functions", "main"])
