"""Lexer, parser and compiler."""

import pytest

from halflife.mite.lexer import tokenize, LexError
from halflife.mite.parser import parse, ParseError
from halflife.mite.compiler import compile_source, CompileError
from halflife.mite.opcodes import Op


def test_tokenize_basics():
    toks = tokenize('let x = 1 + 2; # comment\nprint("hi");')
    kinds = [(t.kind, t.value) for t in toks]
    assert ("kw", "let") in kinds
    assert ("int", 1) in kinds
    assert ("str", "hi") in kinds
    assert kinds[-1][0] == "eof"


def test_tokenize_longest_match():
    toks = tokenize("a == b")
    assert [t.value for t in toks if t.kind == "op"] == ["=="]


def test_string_escapes():
    (tok,) = [t for t in tokenize(r'"a\nb\"c"') if t.kind == "str"]
    assert tok.value == 'a\nb"c'


@pytest.mark.parametrize("src", ['"unterminated', "let x = 1.5;", "x @ y"])
def test_lex_errors(src):
    with pytest.raises(LexError):
        tokenize(src)


def test_parse_precedence():
    prog = parse("let x = 1 + 2 * 3 < 4 and 5 > 6;")
    top = prog.main[0].value
    assert type(top).__name__ == "Logical"
    assert type(top.left).__name__ == "Binary" and top.left.op == "<"
    # 2 * 3 binds tighter than +
    assert top.left.left.right.op == "*"


def test_parse_else_if_chain():
    prog = parse("if (a()) { } else if (b()) { } else { }")
    node = prog.main[0]
    assert type(node.else_body).__name__ == "If"


@pytest.mark.parametrize("src", [
    "let x = ;",
    "if (1) { ",
    "let = 4;",
    "record { a: 1, a: 2 };",
])
def test_parse_errors(src):
    with pytest.raises(ParseError):
        parse(src)


def test_allocation_sites_are_distinct_per_expression():
    m = compile_source(
        "fun f() { return record { a: 1 }; }\n"
        "fun g() { return record { a: 1 }; }\n"
        "let x = f(); let y = g();\n"
    )
    record_sites = [s for s in m.sites if s.kind == "record"]
    assert len(record_sites) == 2
    assert record_sites[0].function == "f" and record_sites[1].function == "g"
    assert record_sites[0].id != record_sites[1].id


def test_shapes_are_interned():
    m = compile_source(
        "fun f() { return record { a: 1, b: 2 }; }\n"
        "fun g() { return record { a: 3, b: 4 }; }\n"
        "let x = f(); let y = g();\n"
    )
    assert m.shapes.count(("a", "b")) == 1


def test_globals_vs_locals():
    m = compile_source("let g = 1;\nfun f() { let l = 2; return l + g; }\nlet z = f();")
    assert m.globals == ["g", "z"]
    f = m.functions[m.func_index["f"]]
    ops = [i[0] for i in f.code]
    assert Op.LOAD_GLOBAL in ops and Op.LOAD_LOCAL in ops


@pytest.mark.parametrize("src,msg", [
    ("let x = y;", "undefined variable"),
    ("f();", "undefined function"),
    ("fun f(a) {} let x = f();", "takes 1 argument"),
    ("return 1;", "outside of a function"),
    ("break;", "break outside loop"),
    ("let a = 1; let a = 2;", "duplicate global"),
    ("fun print() {}", "shadows a builtin"),
])
def test_compile_errors(src, msg):
    with pytest.raises(CompileError) as e:
        compile_source(src)
    assert msg in str(e.value)


def test_disassembler_runs():
    m = compile_source("fun f() { return 1; } let x = f();")
    text = m.disassemble()
    assert "fun f/0" in text and "RET" in text
