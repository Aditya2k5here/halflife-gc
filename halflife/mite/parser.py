"""Recursive-descent parser with precedence climbing for Mite.

Grammar (EBNF-ish)::

    program    := (funcdecl | statement)*
    funcdecl   := "fun" IDENT "(" params ")" block
    statement  := let | assign | if | while | return | break | continue | block | exprstmt
    let        := "let" IDENT "=" expr ";"
    if         := "if" "(" expr ")" block ("else" (if | block))?
    while      := "while" "(" expr ")" block
    expr       := or_expr
    or_expr    := and_expr (("or"|"||") and_expr)*
    and_expr   := equality (("and"|"&&") equality)*
    equality   := comparison (("=="|"!=") comparison)*
    comparison := term (("<"|"<="|">"|">=") term)*
    term       := factor (("+"|"-") factor)*
    factor     := unary (("*"|"/"|"%") unary)*
    unary      := ("-" | "not") unary | postfix
    postfix    := primary ( "(" args ")" | "." IDENT | "[" expr "]" )*
    primary    := INT | STR | "true" | "false" | "nil" | IDENT
                | "(" expr ")" | record | list
    record     := "record" "{" (IDENT ":" expr ("," IDENT ":" expr)*)? "}"
    list       := "[" (expr ("," expr)*)? "]"
"""

from . import ast_nodes as A
from .lexer import tokenize


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens, filename="<mite>"):
        self.toks = tokens
        self.pos = 0
        self.filename = filename

    # --- token helpers -----------------------------------------------------
    @property
    def cur(self):
        return self.toks[self.pos]

    def at(self, kind, value=None):
        t = self.cur
        return t.kind == kind and (value is None or t.value == value)

    def accept(self, kind, value=None):
        if self.at(kind, value):
            t = self.cur
            self.pos += 1
            return t
        return None

    def expect(self, kind, value=None):
        t = self.accept(kind, value)
        if t is None:
            got = self.cur
            raise ParseError(
                "%s:%d:%d: expected %s%s, got %s %r"
                % (
                    self.filename,
                    got.line,
                    got.col,
                    kind,
                    " %r" % value if value else "",
                    got.kind,
                    got.value,
                )
            )
        return t

    # --- entry point -------------------------------------------------------
    def parse_program(self):
        functions = []
        main = []
        while not self.at("eof"):
            if self.at("kw", "fun"):
                functions.append(self.parse_function())
            else:
                main.append(self.parse_statement())
        return A.Program(functions, main, line=1)

    def parse_function(self):
        kw = self.expect("kw", "fun")
        name = self.expect("ident").value
        self.expect("op", "(")
        params = []
        if not self.at("op", ")"):
            while True:
                params.append(self.expect("ident").value)
                if not self.accept("op", ","):
                    break
        self.expect("op", ")")
        body = self.parse_block()
        return A.FunctionDecl(name, params, body, line=kw.line)

    def parse_block(self):
        br = self.expect("op", "{")
        stmts = []
        while not self.at("op", "}"):
            if self.at("eof"):
                raise ParseError("%s:%d: unterminated block" % (self.filename, br.line))
            stmts.append(self.parse_statement())
        self.expect("op", "}")
        return A.Block(stmts, line=br.line)

    # --- statements --------------------------------------------------------
    def parse_statement(self):
        t = self.cur
        if self.at("kw", "let"):
            self.pos += 1
            name = self.expect("ident").value
            self.expect("op", "=")
            value = self.parse_expr()
            self.expect("op", ";")
            return A.Let(name, value, line=t.line)
        if self.at("kw", "if"):
            return self.parse_if()
        if self.at("kw", "while"):
            self.pos += 1
            self.expect("op", "(")
            cond = self.parse_expr()
            self.expect("op", ")")
            body = self.parse_block()
            return A.While(cond, body, line=t.line)
        if self.at("kw", "return"):
            self.pos += 1
            value = None if self.at("op", ";") else self.parse_expr()
            self.expect("op", ";")
            return A.Return(value, line=t.line)
        if self.at("kw", "break"):
            self.pos += 1
            self.expect("op", ";")
            return A.Break(line=t.line)
        if self.at("kw", "continue"):
            self.pos += 1
            self.expect("op", ";")
            return A.Continue(line=t.line)
        if self.at("op", "{"):
            return self.parse_block()

        # expression statement, possibly an assignment
        expr = self.parse_expr()
        if self.accept("op", "="):
            value = self.parse_expr()
            self.expect("op", ";")
            if not isinstance(expr, (A.Name, A.FieldAccess, A.Index)):
                raise ParseError(
                    "%s:%d: invalid assignment target" % (self.filename, t.line)
                )
            return A.Assign(expr, value, line=t.line)
        self.expect("op", ";")
        return A.ExprStmt(expr, line=t.line)

    def parse_if(self):
        t = self.expect("kw", "if")
        self.expect("op", "(")
        cond = self.parse_expr()
        self.expect("op", ")")
        then_body = self.parse_block()
        else_body = None
        if self.accept("kw", "else"):
            else_body = self.parse_if() if self.at("kw", "if") else self.parse_block()
        return A.If(cond, then_body, else_body, line=t.line)

    # --- expressions -------------------------------------------------------
    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.at("kw", "or") or self.at("op", "||"):
            line = self.cur.line
            self.pos += 1
            right = self.parse_and()
            left = A.Logical("or", left, right, line=line)
        return left

    def parse_and(self):
        left = self.parse_equality()
        while self.at("kw", "and") or self.at("op", "&&"):
            line = self.cur.line
            self.pos += 1
            right = self.parse_equality()
            left = A.Logical("and", left, right, line=line)
        return left

    def _binary_level(self, ops, sub):
        left = sub()
        while self.cur.kind == "op" and self.cur.value in ops:
            op = self.cur.value
            line = self.cur.line
            self.pos += 1
            right = sub()
            left = A.Binary(op, left, right, line=line)
        return left

    def parse_equality(self):
        return self._binary_level(("==", "!="), self.parse_comparison)

    def parse_comparison(self):
        return self._binary_level(("<", "<=", ">", ">="), self.parse_term)

    def parse_term(self):
        return self._binary_level(("+", "-"), self.parse_factor)

    def parse_factor(self):
        return self._binary_level(("*", "/", "%"), self.parse_unary)

    def parse_unary(self):
        if self.at("op", "-") or self.at("kw", "not"):
            op = "-" if self.cur.value == "-" else "not"
            line = self.cur.line
            self.pos += 1
            return A.Unary(op, self.parse_unary(), line=line)
        return self.parse_postfix()

    def parse_postfix(self):
        expr = self.parse_primary()
        while True:
            if self.at("op", "("):
                line = self.cur.line
                self.pos += 1
                args = []
                if not self.at("op", ")"):
                    while True:
                        args.append(self.parse_expr())
                        if not self.accept("op", ","):
                            break
                self.expect("op", ")")
                expr = A.Call(expr, args, line=line)
            elif self.at("op", "."):
                line = self.cur.line
                self.pos += 1
                name = self.expect("ident").value
                expr = A.FieldAccess(expr, name, line=line)
            elif self.at("op", "["):
                line = self.cur.line
                self.pos += 1
                idx = self.parse_expr()
                self.expect("op", "]")
                expr = A.Index(expr, idx, line=line)
            else:
                return expr

    def parse_primary(self):
        t = self.cur
        if t.kind == "int":
            self.pos += 1
            return A.IntLit(t.value, line=t.line)
        if t.kind == "str":
            self.pos += 1
            return A.StrLit(t.value, line=t.line)
        if self.accept("kw", "true"):
            return A.BoolLit(True, line=t.line)
        if self.accept("kw", "false"):
            return A.BoolLit(False, line=t.line)
        if self.accept("kw", "nil"):
            return A.NilLit(line=t.line)
        if t.kind == "ident":
            self.pos += 1
            return A.Name(t.value, line=t.line)
        if self.accept("op", "("):
            e = self.parse_expr()
            self.expect("op", ")")
            return e
        if self.at("kw", "record"):
            return self.parse_record()
        if self.at("op", "["):
            return self.parse_list()
        raise ParseError(
            "%s:%d:%d: unexpected %s %r" % (self.filename, t.line, t.col, t.kind, t.value)
        )

    def parse_record(self):
        t = self.expect("kw", "record")
        self.expect("op", "{")
        keys, values = [], []
        if not self.at("op", "}"):
            while True:
                key_tok = self.cur
                if key_tok.kind not in ("ident", "kw"):
                    raise ParseError(
                        "%s:%d: record field name expected" % (self.filename, key_tok.line)
                    )
                self.pos += 1
                key = key_tok.value
                if key in keys:
                    raise ParseError(
                        "%s:%d: duplicate record field %r"
                        % (self.filename, key_tok.line, key)
                    )
                self.expect("op", ":")
                keys.append(key)
                values.append(self.parse_expr())
                if not self.accept("op", ","):
                    break
        self.expect("op", "}")
        return A.RecordLit(keys, values, line=t.line)

    def parse_list(self):
        t = self.expect("op", "[")
        elems = []
        if not self.at("op", "]"):
            while True:
                elems.append(self.parse_expr())
                if not self.accept("op", ","):
                    break
        self.expect("op", "]")
        return A.ListLit(elems, line=t.line)


def parse(src, filename="<mite>"):
    return Parser(tokenize(src, filename), filename).parse_program()
