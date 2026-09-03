"""Hand-written lexer for Mite.

Mite is deliberately tiny. It exists for one reason: to give us a workload
generator where *every allocation has a stable, compiler-assigned identity*.
Using a real language (Python, JS) would mean fighting an allocator we do not
control and cannot attribute; using synthetic random traces would mean the
allocation patterns are made up. A small real language is the honest middle.
"""


class Token:
    __slots__ = ("kind", "value", "line", "col")

    def __init__(self, kind, value, line, col):
        self.kind = kind
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return "Token(%s, %r, line=%d)" % (self.kind, self.value, self.line)


class LexError(Exception):
    pass


KEYWORDS = {
    "let", "fun", "return", "if", "else", "while", "true", "false", "nil",
    "record", "and", "or", "not", "break", "continue",
}

# Longest-first so that ``==`` is not lexed as two ``=``.
OPERATORS = [
    "==", "!=", "<=", ">=", "&&", "||",
    "+", "-", "*", "/", "%", "<", ">", "=",
    "(", ")", "{", "}", "[", "]", ",", ";", ":", ".",
]


def tokenize(src, filename="<mite>"):
    tokens = []
    i = 0
    line = 1
    col = 1
    n = len(src)

    def push(kind, value, ln, cl):
        tokens.append(Token(kind, value, ln, cl))

    while i < n:
        c = src[i]

        # whitespace
        if c == "\n":
            line += 1
            col = 1
            i += 1
            continue
        if c in " \t\r":
            i += 1
            col += 1
            continue

        # comments: # to end of line
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue

        start_line, start_col = line, col

        # numbers (integers only -- Mite has no floats, which keeps every
        # arithmetic result exactly reproducible across machines)
        if c.isdigit():
            j = i
            while j < n and src[j].isdigit():
                j += 1
            if j < n and src[j] == ".":
                raise LexError(
                    "%s:%d:%d: Mite has no floating point numbers"
                    % (filename, line, col)
                )
            push("int", int(src[i:j]), start_line, start_col)
            col += j - i
            i = j
            continue

        # identifiers / keywords
        if c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            word = src[i:j]
            push("kw" if word in KEYWORDS else "ident", word, start_line, start_col)
            col += j - i
            i = j
            continue

        # strings
        if c == '"':
            j = i + 1
            buf = []
            while True:
                if j >= n:
                    raise LexError(
                        "%s:%d:%d: unterminated string" % (filename, start_line, start_col)
                    )
                ch = src[j]
                if ch == "\\":
                    if j + 1 >= n:
                        raise LexError("%s:%d: bad escape" % (filename, line))
                    esc = src[j + 1]
                    buf.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(esc, esc))
                    j += 2
                    continue
                if ch == '"':
                    j += 1
                    break
                if ch == "\n":
                    raise LexError(
                        "%s:%d: newline inside string literal" % (filename, line)
                    )
                buf.append(ch)
                j += 1
            push("str", "".join(buf), start_line, start_col)
            col += j - i
            i = j
            continue

        # operators
        for opstr in OPERATORS:
            if src.startswith(opstr, i):
                push("op", opstr, start_line, start_col)
                i += len(opstr)
                col += len(opstr)
                break
        else:
            raise LexError(
                "%s:%d:%d: unexpected character %r" % (filename, line, col, c)
            )

    push("eof", None, line, col)
    return tokens
