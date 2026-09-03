"""The Mite heap: object model and object store.

Design note (read this before judging the fidelity of the numbers):

This is a **cost simulator**, not a memory-layout simulator. Objects live in a
dict keyed by a monotonically increasing address, and a "copy" does not really
move bytes -- it flips a space tag and adds the object's size to a counter. What
that buys us is that every reported number is an exact, deterministic, machine
independent event count (bytes traced, bytes copied, objects promoted) instead of
a wall-clock number that would mostly measure CPython.

What it costs us is that we model neither fragmentation nor locality. A real
copying collector also wins by compacting the heap and improving cache
behaviour; a real mark-sweep collector also loses to fragmentation. Neither
effect appears here. That cuts both ways and is stated again in the README's
limitations section.
"""

WORD_BYTES = 8
HEADER_WORDS = 1

KIND_RECORD = 0
KIND_LIST = 1
KIND_ARRAY = 2
KIND_STRING = 3

KIND_NAMES = {
    KIND_RECORD: "record",
    KIND_LIST: "list",
    KIND_ARRAY: "array",
    KIND_STRING: "string",
}

# Spaces
SPACE_EDEN = 0     # nursery / young generation
SPACE_OLD = 1      # tenured generation
SPACE_SINGLE = 2   # the one space of a non-generational collector


class Obj:
    """One heap object.

    ``fields`` holds *tagged values* and is the only thing the collector scans.
    ``length`` and ``text`` are raw payload that can never contain a pointer, so
    they are deliberately kept out of the scan path -- exactly the distinction a
    real collector makes between pointer and non-pointer object slots.
    """

    __slots__ = (
        "addr", "kind", "shape", "fields", "text", "length",
        "size_words", "site", "space", "age", "seq", "birth_bytes",
        "marked", "pretenured",
    )

    def __init__(self, addr, kind, size_words, site, seq, birth_bytes):
        self.addr = addr
        self.kind = kind
        self.size_words = size_words
        self.site = site
        self.seq = seq
        self.birth_bytes = birth_bytes
        self.shape = None
        self.fields = []
        self.text = None
        self.length = 0
        self.space = SPACE_EDEN
        self.age = 0
        self.marked = False
        self.pretenured = False

    @property
    def size_bytes(self):
        return self.size_words * WORD_BYTES

    def kind_name(self):
        return KIND_NAMES[self.kind]

    def __repr__(self):
        return "<%s @%d %dB site=%d>" % (
            self.kind_name(), self.addr, self.size_bytes, self.site
        )


def record_words(nfields):
    # header + one slot per field. The field *names* live in the module's
    # interned shape table, not in the object -- that is what a hidden class is.
    return HEADER_WORDS + nfields


def array_words(capacity):
    return HEADER_WORDS + 1 + capacity          # header + capacity word + slots


def list_words():
    return HEADER_WORDS + 2                     # header + length + backing ptr


def string_words(n_chars):
    return HEADER_WORDS + 1 + (n_chars + WORD_BYTES - 1) // WORD_BYTES


class Heap:
    """Flat object store. Policy (where objects go, when to collect) lives in
    the collector; the heap only knows how to make and destroy objects."""

    def __init__(self, record_events=False, max_events=60000):
        self.objects = {}
        self._next_addr = 1
        self.alloc_seq = 0
        self.bytes_allocated = 0
        self.live_bytes = [0, 0, 0]   # indexed by SPACE_*
        self.objects_allocated = 0
        # Optional event tape, used only by the visualiser. Off by default:
        # recording every birth, promotion and death of a few hundred thousand
        # objects costs more memory than the heap being simulated.
        self.events = [] if record_events else None
        self.max_events = max_events

    def _record(self, *event):
        if self.events is not None and len(self.events) < self.max_events:
            self.events.append(event)

    def new(self, kind, size_words, site, space=SPACE_EDEN):
        addr = self._next_addr
        self._next_addr += 1
        obj = Obj(addr, kind, size_words, site, self.alloc_seq, self.bytes_allocated)
        obj.space = space
        self.objects[addr] = obj
        self.alloc_seq += 1
        self.objects_allocated += 1
        nbytes = obj.size_bytes
        self.bytes_allocated += nbytes
        self.live_bytes[space] += nbytes
        self._record("a", obj.seq, nbytes, space, site, self.bytes_allocated)
        return obj

    def free(self, obj):
        self.live_bytes[obj.space] -= obj.size_bytes
        del self.objects[obj.addr]
        self._record("f", obj.seq)

    def move_space(self, obj, space):
        self.live_bytes[obj.space] -= obj.size_bytes
        obj.space = space
        self.live_bytes[space] += obj.size_bytes
        self._record("m", obj.seq, space)

    @property
    def total_live_bytes(self):
        return sum(self.live_bytes)
